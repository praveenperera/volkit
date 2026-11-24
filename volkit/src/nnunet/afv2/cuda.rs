//! The CUDA engine: cuDNN convolutions, cuBLAS GEMMs and volkit's PTX kernels (module docs in [`super`])

use std::path::Path;
use std::time::Instant;

use color_eyre::eyre::{Result, ensure, eyre};
use serde::Serialize;

use super::super::export::Export;
use super::super::{AfvV2Recipe, Contract, Half};
use super::{AfvPrecision, CLASSES, Column, ColumnOut, MAX_BATCH, NONFINITE, TileStats, clip};
use crate::dnn::{Blas, ConvPlan, ConvShape, Dnn, Dtype, Pick, algo_cache};
use crate::gpu::{Arg, Buf, Gpu};
use crate::unet3d::model::{Ctx, Head, HostNet, Model, u32_extent};
use crate::weights::Checkpoint;

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
#[non_exhaustive]
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
    /// A one-line description of each convolution plan, with the time cuDNN measured for it
    pub plans: Vec<String>,
    /// The cuDNN algorithm chosen for each distinct convolution, in forward order (stem, encoder, decoder), then
    /// the float32 seg head's (the bf16 head is a kernel of volkit's own and adds none). Each entry is
    /// `shape | ALGO MATH ws N B`, with ` nondeterministic` or ` (unsupported)` appended where they apply, and
    /// carries no timing, so two loads chose the same cuDNN algorithms exactly when their lists are equal. It covers
    /// the cuDNN choice only: the same choices under another cuDNN or cuBLAS release can still round differently
    /// (the decoder's GEMMs run on cuBLAS), so a caller deciding whether two loads compute the same bits compares
    /// [`cudnn`](Self::cudnn) and [`cublas`](Self::cublas) too
    pub algos: Vec<String>,
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
        let mut algos = model_gpu.algos();
        if let SegHead::Conv { plan, .. } = &head {
            algos.push(format!("{} | {}", plan.shape, plan.algo.choice()));
        }

        let load = LoadReport {
            model: contract.name.clone(),
            precision: opts.precision,
            patch: p,
            border: recipe.border,
            load_s: 0.0,
            plans: model_gpu.plans(),
            algos,
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
        self.finish_into(quantize, Vec::new(), Vec::new())
    }

    /// [`Self::finish`], downloading the outputs into `semantic`'s and `tracing`'s allocations, whatever they held,
    /// so a caller that hands back each finished column's two vectors allocates them about once per run instead of
    /// once per column (about 520 MB a column at nz 1000). A vector without room for its output is replaced by one
    /// sized exactly to it, and vectors the column does not quantize into are dropped
    pub fn finish_into(&mut self, quantize: bool, mut semantic: Vec<u8>, mut tracing: Vec<u8>) -> Result<ColumnOut> {
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
        self.gpu.download_range_into(sem, 0, 2 * n, &mut semantic)?;
        self.gpu.download_range_into(tra, 0, 2 * hn, &mut tracing)?;
        Ok(ColumnOut { shape, semantic: Some(semantic), tracing: Some(tracing), uncovered, tiles: col.tiles })
    }
}
