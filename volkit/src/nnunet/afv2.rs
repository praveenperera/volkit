//! AFV v2's seg inference (`afv-v2` recipe) as villa's `lasagna/qualzz_inference.py` runs it, one output column
//! at a time (inkrs's `inkrs.AfvV2` is this engine)
//!
//! The caller keeps the reference's tiling, reads and air skipping: 128^3 tiles on a lattice of
//! stride 96 over the band padded by 16, reflected at its edges, and grouped into columns of the output. Per
//! column it calls [`AfvV2::begin`], then [`AfvV2::add`] with the column's non-air tiles in the reference's
//! order, then [`AfvV2::finish`]. Per tile the engine runs what the reference runs, in float32:
//! - `normalize_ct`: `(code - mean) / max(std, 10)` with the tile's own mean and unbiased std, each uint8 code
//!   mapped through a 256-entry float32 table computed on the host from the tile's histogram (exact moments in
//!   float64, then the float32 subtract and divide the reference takes);
//! - the network with float32 tensors and TF32 convolution and GEMM math, as PyTorch's float32 defaults run it,
//!   then the 1x1x1 seg head on cuDNN with its bias added on its own;
//! - the float32 softmax of the four classes, vertical and horizontal kept, blended into column accumulators with
//!   the tile's flat ramp weight (0 on the 16-voxel border, 1 on the core), each product and sum rounded on its
//!   own.
//!
//! [`AfvV2::finish`] normalises the accumulators by their weight and returns the `semantic` uint8 output and the
//! 2x2x2-pooled `tracing` one, as the reference's finalise computes them. The centre, affinity and direction heads
//! are not run: the tracer reads only `semantic` and `tracing` (villa `trace_fibers.py`)
//!
//! [`AfvPrecision::Bf16`] is an opt-in that changes the output: the network and the seg head run in bf16 as
//! PyTorch's bf16 autocast runs them (the input cast to bf16, convs in bf16 with float32 accumulation, every
//! conv output, bias add, norm, residual add and LeakyReLU rounded to bf16, the head's logits rounded to bf16 and
//! then cast to float32), and everything from the softmax on is the float32 path's

use std::path::Path;
use std::time::Instant;

use color_eyre::eyre::{Result, ensure, eyre};
use serde::Serialize;

use super::export::Export;
use super::{AfvV2Recipe, Contract, Half, Precision};
use crate::dnn::{Blas, ConvPlan, ConvShape, Dnn, Dtype, Pick, algo_cache};
use crate::gpu::{Arg, Buf, Gpu};
use crate::unet3d::model::{Ctx, Head, HostNet, Model, u32_extent};
use crate::weights::Checkpoint;

/// `af_finish` / `af_pool`'s status bits (volkit-kernels `afv2.rs`)
const NONFINITE: u32 = 1;
/// The seg head's classes: background, vertical, horizontal, intersection
const CLASSES: usize = 4;
/// Tiles one [`AfvV2::add`] call takes at most (the reference batches 4)
pub const MAX_BATCH: usize = 16;

/// The arithmetic the network runs in
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AfvPrecision {
    /// The default, for parity with the reference runner (villa's `lasagna.qualzz_inference` on PyTorch): float32
    /// tensors with TF32 tensor-core convolution and GEMM math, as PyTorch's float32 defaults run the model
    #[default]
    Fp32,
    /// Opt-in speed mode with small, measured differences: PyTorch's bf16 autocast forward (see the module docs).
    /// Measured against float32 on two CT bands, about 99% of traced fibre length agrees (99.2-99.4%), 97.0-97.8%
    /// of fibres match and fibre counts are within 0.3%. Never the default
    Bf16,
}

impl AfvPrecision {
    /// The precision by its name, `fp32` or `bf16`
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "fp32" => Ok(Self::Fp32),
            "bf16" => Ok(Self::Bf16),
            _ => Err(eyre!("precision {name:?}; AfvV2 runs fp32 (the reference contract) or bf16")),
        }
    }

    /// The forward's storage and math
    fn net(self) -> Precision {
        match self {
            Self::Fp32 => Precision::Tf32,
            Self::Bf16 => Precision::Bf16,
        }
    }
}

/// Engine options
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// CUDA device ordinal
    pub device: usize,
    /// How the engine chooses cuDNN algorithms
    pub pick: Pick,
    /// The precision the forward runs in
    pub precision: AfvPrecision,
}

/// What loading the engine took and chose
#[derive(Clone, Debug, Serialize)]
pub struct LoadReport {
    /// The model path or name loaded
    pub model: String,
    /// The precision the forward runs in
    pub precision: AfvPrecision,
    /// Tile edge length in voxels
    pub patch: usize,
    /// Tile border the blend ramp zeroes, in voxels
    pub border: usize,
    /// Seconds spent loading and planning
    pub load_s: f64,
    /// A one-line description of each convolution plan
    pub plans: Vec<String>,
    /// The seg head's cuDNN plan (fp32) or kernel (bf16), and the output channels it runs (4, or padded to 8 on
    /// cuDNN)
    pub head_plan: String,
    /// Output channels the seg head runs
    pub head_channels: usize,
    /// State of the cuDNN algorithm cache for this load
    pub algo_cache: algo_cache::Status,
    /// The cuDNN this process loaded (`cudnnGetVersion`): inside PyTorch it is PyTorch's own, which can choose
    /// other kernels than the system cuDNN a standalone process loads, and so other output bits
    pub cudnn: usize,
    /// The cuBLAS this process loaded (`cublasGetVersion`, such as 120804 for 12.8.4). volkit calls it through CUDA
    /// 12 bindings, so anything but 12.x means the loader found another major first (see the README's library
    /// versions note)
    pub cublas: i32,
    /// Whether the stem's input channels are padded because cuDNN refuses a one-channel 3D convolution
    pub stem_padded: bool,
    /// The PTX tier whose kernels were loaded
    pub ptx_tier: String,
    /// The device's name
    pub gpu: String,
    /// Floating-point operations per tile
    pub flops_per_tile: u64,
}

/// One column's outputs: `semantic` `(2, nz, ny, nx)` and `tracing` `(2, nz / 2, ny / 2, nx / 2)` uint8, channels
/// vertical then horizontal, or `None` when no tile ran (the reference writes nothing for an all-air column)
#[derive(Debug)]
pub struct ColumnOut {
    /// The column's `[z, y, x]` size in voxels
    pub shape: [usize; 3],
    /// The semantic output, or `None` when no tile ran
    pub semantic: Option<Vec<u8>>,
    /// The tracing output, or `None` when no tile ran
    pub tracing: Option<Vec<u8>>,
    /// Voxels no tile weighted (the reference's `voxels_uncovered`)
    pub uncovered: u64,
    /// Tiles the column ran
    pub tiles: usize,
}

/// A tile's normalisation: the reference's float32 mean and floored unbiased std
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct TileStats {
    /// The tile's mean over its codes
    pub mean: f32,
    /// The tile's unbiased standard deviation, floored
    pub std: f32,
}

impl TileStats {
    /// The statistics of `tile`'s codes, the mean and the unbiased std floored at `floor`, from exact float64
    /// moments of its histogram
    pub fn of(tile: &[u8], floor: f32) -> Result<Self> {
        ensure!(tile.len() > 1, "a tile of {} voxels has no unbiased std", tile.len());
        let mut hist = [[0u64; 256]; 4];
        let (quads, rest) = tile.as_chunks::<4>();
        for c in quads {
            for (h, &v) in hist.iter_mut().zip(c) {
                h[v as usize] += 1;
            }
        }
        for &v in rest {
            hist[0][v as usize] += 1;
        }
        let counts: Vec<u64> = (0..256).map(|i| hist.iter().map(|h| h[i]).sum()).collect();
        let n = tile.len() as f64;
        // the sum of codes is an integer below 2^53, exact in float64
        let sum: u64 = counts.iter().enumerate().map(|(code, &c)| code as u64 * c).sum();
        let mean = sum as f64 / n;
        let ss: f64 = counts.iter().enumerate().map(|(code, &c)| c as f64 * (code as f64 - mean).powi(2)).sum();
        let std = ((ss / (n - 1.0)).sqrt() as f32).max(floor);
        Ok(Self { mean: mean as f32, std })
    }

    /// `(code - mean) / std` for every code, each a float32 operation as the reference's `(ct - mean) / std`
    pub fn table(&self) -> [f32; 256] {
        std::array::from_fn(|code| (code as f32 - self.mean) / self.std)
    }
}

/// The column being accumulated
struct Column {
    shape: [usize; 3],
    y0: usize,
    x0: usize,
    tiles: usize,
}

/// The seg head, writing four float32 logits per tile voxel into `AfvV2::logits`
enum SegHead {
    /// fp32: a cuDNN 1x1x1 conv with TF32 math, weight `[k, cin]` (rows past the four classes zero), into
    /// `[p^3, k]`; `af_fold` adds the float32 bias `b` (the plan is boxed: it is most of the enum's size)
    Conv { plan: Box<ConvPlan>, w: Buf, b: Buf, k: usize, ws: Buf },
    /// bf16: `nf_head` on the model's bf16 head weights, into `[4, p^3]` with the bias added in bf16; `pair` is the
    /// kernel's scratch for flip pairs, unused at one flip
    Bf16 { pair: Buf },
}

/// An AFV v2 export on the GPU, accumulating one column at a time
pub struct AfvV2 {
    // field order is drop order: buffers and plans, then the handles, then the GPU's context
    model: Model,
    head: SegHead,
    logits: Buf,
    ramp: Buf,
    /// A batch of tiles, `[MAX_BATCH, p^3]` uint8, and their tables `[MAX_BATCH, 256]`: float32, or bf16 bits in
    /// the first half of each row in bf16 mode
    tiles: Buf,
    tables: Buf,
    stat: Buf,
    /// The column's accumulators `[2, n]` and `[n]` float32 and its outputs; grown on demand
    acc: Option<Buf>,
    wsum: Option<Buf>,
    sem: Option<Buf>,
    tra: Option<Buf>,
    col: Option<Column>,
    blas: Blas,
    dnn: Dnn,
    gpu: Gpu,
    pick: Pick,
    precision: AfvPrecision,
    recipe: AfvV2Recipe,
    /// What loading the engine took and chose
    pub load: LoadReport,
}

impl Drop for AfvV2 {
    fn drop(&mut self) {
        // the buffers' and handles' destructors need the context current on the thread the last reference drops
        // on (a Python finalizer, say)
        let _ = self.gpu.inner.context().bind_to_thread();
    }
}

// SAFETY: every method binds the engine's context first, `&mut self` keeps two threads from using it at once, and
// `Gpu`'s kernel cache is reached only through those methods
unsafe impl Send for AfvV2 {}

/// A growable device buffer of at least `bytes`
fn ensure_buf<'a>(gpu: &Gpu, slot: &'a mut Option<Buf>, bytes: usize) -> Result<&'a Buf> {
    if slot.as_ref().is_none_or(|b| b.bytes() < bytes) {
        *slot = None;
        *slot = Some(gpu.alloc(bytes)?);
    }
    slot.as_ref().ok_or_else(|| eyre!("buffer not allocated"))
}

/// Zeroes the first `bytes` of `buf`, queued on the stream
fn zero(gpu: &Gpu, buf: &Buf, bytes: usize) -> Result<()> {
    ensure!(bytes <= buf.bytes(), "zero range past the end of the buffer");
    // SAFETY: the range lies inside the live allocation, checked above
    unsafe { cudarc::driver::result::memset_d8_async(buf.ptr(), 0, bytes, gpu.cu_stream()) }?;
    Ok(())
}

/// The tile voxels `[lo, hi)` on one axis whose volume coordinates `origin - border + i` fall in the column's
/// `[c0, c0 + len)`, and the column coordinate of the first: the reference `add`'s clip
fn clip(origin: usize, border: usize, c0: usize, len: usize, p: usize) -> Option<(usize, usize, usize)> {
    // tile voxel i sits at volume coordinate origin - border + i, column coordinate that minus c0
    let lo = origin as i64 - border as i64 - c0 as i64;
    let (a, b) = (lo.max(0), (lo + p as i64).min(len as i64));
    (b > a).then(|| ((a - lo) as usize, (b - lo) as usize, a as usize))
}

impl AfvV2 {
    /// Loads the model at `model`, opens the GPU and plans the forward and the seg head. `model` is the export
    /// (`.safetensors` with its `.json` sidecar, `afv-v2` recipe) or the nnU-Net model folder, converted in memory
    /// as [`Export::afv_v2`] converts it
    pub fn open(model: &Path, opts: &Options) -> Result<Self> {
        let t = Instant::now();
        let (ck, sidecar) = if model.is_dir() {
            let (tensors, aliases, sidecar) = Export::afv_v2(model)?.into_parts();
            (Checkpoint::with_sidecar(tensors, aliases, &sidecar)?, sidecar)
        } else {
            Checkpoint::read(model)?
        };
        let contract = Contract::from_sidecar(&sidecar)?;
        let recipe = contract.afv_v2()?;
        let p = contract.patch_size[0];
        let host = HostNet::take(&ck, &contract.program, opts.precision.net())?;
        ck.finish()?;

        let gpu = Gpu::open_device(opts.device)?;
        let dnn = Dnn::new(&gpu)?;
        let blas = Blas::new(&gpu)?;
        let max_workspace = 1 << 30;
        let ctx = Ctx { gpu: &gpu, dnn: &dnn, blas: &blas, pick: opts.pick, max_workspace };
        let model_gpu = Model::new(&ctx, &host, &contract.program, p)?;
        let h = model_gpu.heads();
        ensure!(h.classes == CLASSES, "{} head classes, AFV v2's seg head has {CLASSES}", h.classes);
        let vox = p.pow(3);
        u32_extent("tile", vox * 8)?;

        let (head, head_plan, head_channels) = match opts.precision {
            AfvPrecision::Fp32 => Self::conv_head(&gpu, &dnn, &h, p, opts.pick, max_workspace)?,
            AfvPrecision::Bf16 => {
                let plan = "nf_head: bf16 1x1x1 conv, float32 accumulation, bias added in bf16".to_string();
                (SegHead::Bf16 { pair: gpu.alloc(CLASSES * vox * 4)? }, plan, CLASSES)
            }
        };
        let logits_bytes = vox * head_channels * 4;

        let load = LoadReport {
            model: contract.name.clone(),
            precision: opts.precision,
            patch: p,
            border: recipe.border,
            load_s: 0.0,
            plans: model_gpu.plans(),
            head_plan,
            head_channels,
            algo_cache: algo_cache::status(),
            cudnn: Dnn::version(),
            cublas: gpu.inner.library_versions()?.cublas,
            stem_padded: model_gpu.in_c != 1,
            ptx_tier: gpu.inner.ptx_tier().to_string(),
            gpu: gpu.inner.context().name()?,
            flops_per_tile: contract.flops_per_patch,
        };
        let mut net = Self {
            logits: gpu.alloc(logits_bytes)?,
            ramp: gpu.upload(&recipe.ramp)?,
            tiles: gpu.alloc(MAX_BATCH * vox)?,
            tables: gpu.alloc(MAX_BATCH * 256 * 4)?,
            stat: gpu.alloc_fill(2 * 4, 0)?,
            acc: None,
            wsum: None,
            sem: None,
            tra: None,
            col: None,
            head,
            model: model_gpu,
            blas,
            dnn,
            gpu,
            pick: opts.pick,
            precision: opts.precision,
            recipe,
            load,
        };
        net.gpu.sync()?;
        net.load.load_s = t.elapsed().as_secs_f64();
        Ok(net)
    }

    /// The fp32 contract's seg head as PyTorch runs a float32 1x1x1 conv: cuDNN with TF32 allowed, the bias added
    /// after (by `af_fold`); its plan and the output channels it runs
    fn conv_head(
        gpu: &Gpu,
        dnn: &Dnn,
        h: &Head<'_>,
        p: usize,
        pick: Pick,
        max_workspace: usize,
    ) -> Result<(SegHead, String, usize)> {
        let w: Vec<f32> = gpu.download(h.w)?;
        let b: Vec<f32> = gpu.download(h.b)?;
        ensure!(w.len() == CLASSES * h.cin && b.len() == CLASSES, "seg head of {} weights", w.len());
        let dtype = Dtype::Float { tf32: true };
        let plan = |k: usize| {
            let shape = ConvShape::same(1, h.cin as i32, k as i32, &[p as i32; 3], 1, 1);
            ConvPlan::typed(gpu, dnn, shape, dtype, pick, max_workspace)
        };
        // cuDNN may refuse four output channels; zero rows past the classes leave the four logits unchanged
        let (plan, k) = match plan(CLASSES) {
            Ok(pl) => (pl, CLASSES),
            Err(e) => {
                eprintln!("afv-v2: seg head with {CLASSES} outputs: {e:#}; padding to 8");
                (plan(8)?, 8)
            }
        };
        let mut wk = vec![0f32; k * h.cin];
        wk[..w.len()].copy_from_slice(&w);
        let head_plan = format!("{} | {}", plan.shape, plan.algo);
        let ws = gpu.alloc(plan.workspace().max(1))?;
        Ok((SegHead::Conv { plan: Box::new(plan), w: gpu.upload(&wk)?, b: gpu.upload(&b)?, k, ws }, head_plan, k))
    }

    /// The arithmetic the network runs in
    pub fn precision(&self) -> AfvPrecision {
        self.precision
    }

    /// The tile edge
    pub fn patch(&self) -> usize {
        self.model.patch
    }

    /// The tile border the ramp weights zero
    pub fn border(&self) -> usize {
        self.recipe.border
    }

    /// The normalisation's floor on the std
    pub fn std_floor(&self) -> f32 {
        self.recipe.std_floor
    }

    /// Starts the column of `shape` `(nz, ny, nx)` whose first voxel is at volume `(0, y0, x0)` of the band, with
    /// zeroed accumulators; a column already begun and not finished is dropped
    pub fn begin(&mut self, shape: [usize; 3], y0: usize, x0: usize) -> Result<()> {
        self.gpu.inner.context().bind_to_thread()?;
        let n: usize = shape.iter().product();
        ensure!(n > 0, "empty column {shape:?}");
        u32_extent("column accumulators", 2 * n)?;
        let acc = ensure_buf(&self.gpu, &mut self.acc, 2 * n * 4)?;
        zero(&self.gpu, acc, 2 * n * 4)?;
        let wsum = ensure_buf(&self.gpu, &mut self.wsum, n * 4)?;
        zero(&self.gpu, wsum, n * 4)?;
        self.gpu.fill(&self.stat, 0)?;
        self.col = Some(Column { shape, y0, x0, tiles: 0 });
        Ok(())
    }

    /// Runs `origins.len()` tiles and blends them into the column, in order: `tiles` holds their `p^3` uint8
    /// inputs one after another (the band read and reflected as the reference's `read_tile` does), `origins`
    /// their padded `(pz, py, px)` lattice origins. At most [`MAX_BATCH`] tiles a call; the returned statistics
    /// are each tile's normalisation
    pub fn add(&mut self, tiles: &[u8], origins: &[[usize; 3]]) -> Result<Vec<TileStats>> {
        self.gpu.inner.context().bind_to_thread()?;
        let p = self.model.patch;
        let vox = p.pow(3);
        let count = origins.len();
        ensure!(count <= MAX_BATCH, "{count} tiles in one call; at most {MAX_BATCH}");
        ensure!(tiles.len() == count * vox, "{} bytes for {count} tiles of {vox}", tiles.len());
        let col = self.col.as_ref().ok_or_else(|| eyre!("add before begin"))?;
        let (shape, y0, x0) = (col.shape, col.y0, col.x0);
        if count == 0 {
            return Ok(Vec::new());
        }

        // the statistics on the host, while the device runs the previous batch
        let floor = self.recipe.std_floor;
        let stats: Vec<TileStats> = tiles.chunks_exact(vox).map(|t| TileStats::of(t, floor)).collect::<Result<_>>()?;
        self.gpu.upload_into(&self.tiles, 0, tiles)?;
        match self.precision {
            AfvPrecision::Fp32 => {
                let tables: Vec<f32> = stats.iter().flat_map(|s| s.table()).collect();
                self.gpu.upload_into(&self.tables, 0, &tables)?;
            }
            AfvPrecision::Bf16 => {
                // autocast casts the float32 normalised input to bf16 at the first conv; tile i's table is row i
                let tables: Vec<u16> = stats
                    .iter()
                    .flat_map(|s| s.table().map(|v| Half::Bf16.bits(v)).into_iter().chain([0; 256]))
                    .collect();
                self.gpu.upload_into(&self.tables, 0, &tables)?;
            }
        }

        let border = self.recipe.border;
        let n: usize = shape.iter().product();
        for (i, o) in origins.iter().enumerate() {
            let z = clip(o[0], border, 0, shape[0], p);
            let y = clip(o[1], border, y0, shape[1], p);
            let x = clip(o[2], border, x0, shape[2], p);
            // the reference's `add` returns before blending a tile that misses the column; its forward still ran
            self.forward(i)?;
            let (Some(z), Some(y), Some(x)) = (z, y, x) else { continue };
            let extent = [z.1 - z.0, y.1 - y.0, x.1 - x.0];
            let acc = self.acc.as_ref().ok_or_else(|| eyre!("no accumulators"))?;
            let wsum = self.wsum.as_ref().ok_or_else(|| eyre!("no accumulators"))?;
            // fp32 logits are the conv's interleaved `[p^3, k]` before the bias; bf16's are `nf_head`'s planar
            // `[4, p^3]` with the bias in
            let (bias, vstride, cstride) = match &self.head {
                SegHead::Conv { b, k, .. } => (b.arg::<f32>(), *k, 1),
                SegHead::Bf16 { .. } => (Arg::EMPTY, 1, vox),
            };
            let mut args = vec![
                acc.view::<f32>(0, 2 * n),
                wsum.view::<f32>(0, n),
                self.logits.arg::<f32>(),
                bias,
                self.ramp.arg::<f32>(),
            ];
            args.extend(
                [
                    p, vstride, cstride, z.0, y.0, x.0, extent[0], extent[1], extent[2], z.2, y.2, x.2, shape[1],
                    shape[2],
                ]
                .map(|v| Arg::U32(v as u32)),
            );
            self.gpu.launch_1d("af_fold", extent.iter().product(), &args)?;
        }
        if let Some(c) = self.col.as_mut() {
            c.tiles += count;
        }
        Ok(stats)
    }

    /// The forward and seg head logits of uploaded tile `i`
    fn forward(&mut self, i: usize) -> Result<()> {
        let p = self.model.patch;
        let vox = p.pow(3);
        let ctx = Ctx { gpu: &self.gpu, dnn: &self.dnn, blas: &self.blas, pick: self.pick, max_workspace: 1 << 30 };
        let tile = self.tiles.view::<u8>(i * vox, vox);
        let in_c = Arg::U32(self.model.in_c as u32);
        match self.precision {
            AfvPrecision::Fp32 => {
                let table = self.tables.view::<f32>(i * 256, 256);
                let args = [self.model.input.arg::<f32>(), tile, table, Arg::U32(vox as u32), in_c];
                self.gpu.launch_1d("af_prep", vox, &args)?;
            }
            AfvPrecision::Bf16 => {
                // the tile is a p^3 slab, read whole: corner 0, every voxel inside, no flip
                let table = self.tables.view::<u16>(i * 512, 256);
                let mut args = vec![self.model.input.arg::<u16>(), tile, table];
                args.extend([p, p, 0, 0, 0, p, 0, p, 0, p, 0, p].map(|v| Arg::U32(v as u32)));
                args.extend([in_c, Arg::U32(0)]);
                self.gpu.launch_1d("nn_prep", vox, &args)?;
            }
        }
        self.model.forward(&ctx)?;
        let h = self.model.heads();
        match &self.head {
            SegHead::Conv { plan, w, ws, .. } => {
                let x = (h.x.ptr(), vox * h.cin * 4);
                let y = (self.logits.ptr(), self.logits.bytes());
                plan.run(&self.dnn, x, (w.ptr(), w.bytes()), y, Some(ws))
            }
            SegHead::Bf16 { pair } => {
                let args = [
                    self.logits.arg::<f32>(),
                    pair.arg::<f32>(),
                    h.x.view::<u32>(0, vox * h.cin / 2),
                    h.w.arg::<f32>(),
                    h.b.arg::<f32>(),
                    Arg::U32(h.cin as u32),
                    Arg::U32(h.classes as u32),
                    Arg::U32(p as u32),
                    // no flip, and `NF_HEAD_SET`: the logits are this forward's alone
                    Arg::U32(0),
                    Arg::U32(0),
                ];
                self.gpu.launch_1d("nf_head", vox, &args)
            }
        }
    }

    /// Normalises the column and returns its outputs, or only its uncovered count when `quantize` is false (the
    /// reference writes nothing for a column whose tiles were all air). Fails on a non-finite probability, as the
    /// reference's `quantize` does
    pub fn finish(&mut self, quantize: bool) -> Result<ColumnOut> {
        self.gpu.inner.context().bind_to_thread()?;
        let col = self.col.take().ok_or_else(|| eyre!("finish before begin"))?;
        let shape = col.shape;
        let n: usize = shape.iter().product();
        let acc = self.acc.as_ref().ok_or_else(|| eyre!("no accumulators"))?;
        let wsum = self.wsum.as_ref().ok_or_else(|| eyre!("no accumulators"))?;
        let sem = ensure_buf(&self.gpu, &mut self.sem, 2 * n)?;
        let blocks = n.div_ceil(256) as u32;
        let args =
            [sem.view::<u8>(0, 2 * n), acc.view::<f32>(0, 2 * n), self.stat.view::<u32>(0, 2), wsum.view::<f32>(0, n)];
        self.gpu.launch("af_finish", (blocks, 1, 1), (256, 1, 1), 0, &args)?;
        let half = shape.map(|v| v / 2);
        let hn: usize = half.iter().product();
        let tra = ensure_buf(&self.gpu, &mut self.tra, (2 * hn).max(1))?;
        if hn > 0 {
            let mut args = vec![tra.view::<u8>(0, 2 * hn), self.stat.view::<u32>(0, 2), acc.view::<f32>(0, 2 * n)];
            args.extend(shape.map(|v| Arg::U32(v as u32)));
            self.gpu.launch_1d("af_pool", 2 * hn, &args)?;
        }
        let status: Vec<u32> = self.gpu.download_range(&self.stat, 0, 2)?;
        let uncovered = u64::from(status[1]);
        if !quantize {
            return Ok(ColumnOut { shape, semantic: None, tracing: None, uncovered, tiles: col.tiles });
        }
        ensure!(status[0] & NONFINITE == 0, "non-finite probabilities");
        let semantic = self.gpu.download_range::<u8>(sem, 0, 2 * n)?;
        let tracing = self.gpu.download_range::<u8>(tra, 0, 2 * hn)?;
        Ok(ColumnOut { shape, semantic: Some(semantic), tracing: Some(tracing), uncovered, tiles: col.tiles })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tile statistics are the exact moments: the mean of 0..=255 repeated, the unbiased std, the floor
    #[test]
    fn tile_stats_are_exact_and_floored() {
        let tile: Vec<u8> = (0..4096u32).map(|i| (i % 256) as u8).collect();
        let s = TileStats::of(&tile, 10.0).expect("stats");
        assert_eq!(s.mean, 127.5);
        let var = (0..4096).map(|i| ((i % 256) as f64 - 127.5).powi(2)).sum::<f64>() / 4095.0;
        assert_eq!(s.std, var.sqrt() as f32);
        let flat = TileStats::of(&[7u8; 64], 10.0).expect("stats");
        assert_eq!((flat.mean, flat.std), (7.0, 10.0));
        // the table is the float32 subtract then divide, code by code
        let t = s.table();
        assert_eq!(t[3], (3.0f32 - 127.5) / s.std);
        assert!(TileStats::of(&[1u8], 10.0).is_err());
    }

    /// A tile's clip to the column is the reference `add`'s: the border hangs off the band's start, the shifted
    /// last tile overlaps, and a tile beside the column misses it
    #[test]
    fn clip_matches_the_reference_add() {
        // padded origin 0: tile voxel 16 is volume 0
        assert_eq!(clip(0, 16, 0, 512, 128), Some((16, 128, 0)));
        // column [480, 960): the tile at padded origin 480 starts at volume 464
        assert_eq!(clip(480, 16, 480, 480, 128), Some((16, 128, 0)));
        assert_eq!(clip(384, 16, 480, 480, 128), Some((112, 128, 0)));
        assert_eq!(clip(1200, 16, 480, 480, 128), None);
        // a 9-voxel column at the volume's end
        assert_eq!(clip(8073, 16, 8160, 9, 128), Some((103, 112, 0)));
    }
}
