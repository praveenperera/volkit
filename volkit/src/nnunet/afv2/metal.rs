//! The Metal engine on Apple silicon: volkit's MSL kernels, the convolutions on Metal 4 tensor ops with float32
//! operands (module docs in [`super`])
//!
//! The float32 contract is the CUDA engine's, with one difference in the arithmetic: CUDA runs the convolutions
//! and GEMMs with TF32 operands, as PyTorch's float32 defaults do on NVIDIA GPUs, and Metal runs them with true
//! float32 operands, so its output sits inside TF32's rounding noise of CUDA's. The bf16 opt-in is CUDA-only

use std::path::Path;
use std::time::Instant;

use color_eyre::eyre::{Result, ensure, eyre};
use serde::Serialize;

use super::super::export::Export;
use super::super::{AfvV2Recipe, Contract};
use super::{AfvPrecision, CLASSES, Column, ColumnOut, MAX_BATCH, NONFINITE, TileStats, clip};
use crate::ensure_supported;
use crate::metal::gpu::{Arg, Buf, Gpu};
use crate::metal::model::{Layout, Model, winograd_levels};
use crate::unet3d::host::HostNet;
use crate::weights::Checkpoint;

/// Engine options
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Metal device ordinal: Apple silicon has one GPU, device 0
    pub device: usize,
    /// The precision the forward runs in; only [`AfvPrecision::Fp32`] runs on Metal
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
    /// Seconds spent loading the weights and compiling the kernels
    pub load_s: f64,
    /// The device's name, such as "Apple M4 Pro"
    pub gpu: String,
    /// Bytes the model's weights and activation buffers hold on the device
    pub model_bytes: usize,
    /// The device's recommended working set (`recommendedMaxWorkingSetSize`), in bytes: the most this process should
    /// keep allocated on the GPU before Metal starts paging
    pub working_set: usize,
    /// The largest buffer the device allocates (`maxBufferLength`), in bytes. A column's accumulator is one buffer of
    /// 8 bytes a voxel, so a column of more voxels than an eighth of this fails to begin
    pub max_buffer: usize,
    /// Floating-point operations per tile
    pub flops_per_tile: u64,
    /// The U-Net levels whose stride-1 3x3x3 convolutions run as Winograd F(4,3) (`VOLKIT_METAL_WINOGRAD`), ascending;
    /// their rounding differs from the direct convolution's, so outputs match byte for byte only between equal lists
    pub winograd_levels: Vec<usize>,
}

/// An AFV v2 export on the GPU, accumulating one column at a time
pub struct AfvV2 {
    // field order is drop order: buffers first, then the device, whose drop waits for queued kernels
    model: Model,
    ramp: Buf,
    /// A batch of tiles, `[MAX_BATCH, p^3]` uint8, and their float32 tables `[MAX_BATCH, 256]`
    tiles: Buf,
    tables: Buf,
    /// `af_finish` / `af_pool`'s status words: the non-finite flag and the uncovered count
    stat: Buf,
    /// The column's accumulators `[2, n]` and `[n]` float32 and its outputs; grown on demand
    acc: Option<Buf>,
    wsum: Option<Buf>,
    sem: Option<Buf>,
    tra: Option<Buf>,
    col: Option<Column>,
    gpu: Gpu,
    recipe: AfvV2Recipe,
    /// What loading the engine took and chose
    pub load: LoadReport,
}

/// A growable device buffer of at least `bytes`
fn ensure_buf<'a>(gpu: &Gpu, slot: &'a mut Option<Buf>, bytes: usize) -> Result<&'a Buf> {
    if slot.as_ref().is_none_or(|b| b.bytes() < bytes) {
        *slot = None;
        *slot = Some(gpu.alloc(bytes)?);
    }
    slot.as_ref().ok_or_else(|| eyre!("buffer not allocated"))
}

/// Zeroes the first `words` 32-bit words of `buf`, queued in order with the kernels
fn zero(gpu: &Gpu, buf: &Buf, words: usize) -> Result<()> {
    ensure!(words * 4 <= buf.bytes(), "zero range past the end of the buffer");
    gpu.dispatch_1d("fill_u32", words, &[buf.arg(), Arg::U32(0), words.into()])
}

impl AfvV2 {
    /// Loads the model at `model`, opens the GPU and uploads the forward. `model` is the export (`.safetensors`
    /// with its `.json` sidecar, `afv-v2` recipe) or the nnU-Net model folder, converted in memory as
    /// [`Export::afv_v2`] converts it
    pub fn open(model: &Path, opts: &Options) -> Result<Self> {
        let t = Instant::now();
        ensure_supported!(
            opts.precision == AfvPrecision::Fp32,
            "the Metal engine runs fp32 only; the bf16 opt-in is CUDA-only"
        );
        let (ck, sidecar) = if model.is_dir() {
            let (tensors, aliases, sidecar) = Export::afv_v2(model)?.into_parts();
            (Checkpoint::with_sidecar(tensors, aliases, &sidecar)?, sidecar)
        } else {
            Checkpoint::read(model)?
        };
        let contract = Contract::from_sidecar(&sidecar)?;
        let recipe = contract.afv_v2()?;
        let p = contract.patch_size[0];
        let vox = p.pow(3);
        ensure_supported!(u32::try_from(vox * 64).is_ok(), "a {p}^3 tile overflows 32-bit indexing");
        let layout = Layout::new(&contract.program, p, &winograd_levels()?)?;
        // the device only once the model has parsed and its shape checked, so a bad or unsupported model fails
        // without touching the GPU; the kernel library then compiles on its own thread while the weights are laid out
        let gpu = Gpu::open(opts.device)?;
        let host = HostNet::take(&ck, &contract.program, opts.precision.net())?;
        ck.finish()?;
        ensure!(host.head_b.len() == CLASSES, "{} head classes, AFV v2's seg head has {CLASSES}", host.head_b.len());

        let model_gpu = Model::new(&gpu, &host, &layout)?;
        gpu.ready()?;
        let load = LoadReport {
            model: contract.name.clone(),
            precision: opts.precision,
            patch: p,
            border: recipe.border,
            load_s: 0.0,
            gpu: gpu.name(),
            model_bytes: model_gpu.bytes(),
            working_set: gpu.memory().1,
            max_buffer: gpu.max_buffer(),
            flops_per_tile: contract.flops_per_patch,
            winograd_levels: model_gpu.winograd.clone(),
        };
        let mut net = Self {
            ramp: gpu.upload(&recipe.ramp)?,
            tiles: gpu.alloc(MAX_BATCH * vox)?,
            tables: gpu.alloc(MAX_BATCH * 256 * 4)?,
            stat: gpu.alloc(2 * 4)?,
            acc: None,
            wsum: None,
            sem: None,
            tra: None,
            col: None,
            model: model_gpu,
            gpu,
            recipe,
            load,
        };
        net.load.load_s = t.elapsed().as_secs_f64();
        Ok(net)
    }

    /// The arithmetic the network runs in
    pub fn precision(&self) -> AfvPrecision {
        AfvPrecision::Fp32
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
        self.col = None;
        let n: usize = shape.iter().product();
        ensure!(n > 0, "empty column {shape:?}");
        ensure_supported!(u32::try_from(2 * n).is_ok(), "a column of {n} voxels overflows 32-bit indexing");
        let acc = ensure_buf(&self.gpu, &mut self.acc, 2 * n * 4)?;
        zero(&self.gpu, acc, 2 * n)?;
        let wsum = ensure_buf(&self.gpu, &mut self.wsum, n * 4)?;
        zero(&self.gpu, wsum, n)?;
        zero(&self.gpu, &self.stat, 2)?;
        self.col = Some(Column { shape, y0, x0, tiles: 0 });
        Ok(())
    }

    /// Runs `origins.len()` tiles and blends them into the column, in order: `tiles` holds their `p^3` uint8
    /// inputs one after another (the band read and reflected as the reference's `read_tile` does), `origins`
    /// their padded `(pz, py, px)` lattice origins. At most [`MAX_BATCH`] tiles a call; the returned statistics
    /// are each tile's normalisation
    ///
    /// A failure drops the column, so a later [`Self::add`] or [`Self::finish`] fails until the next
    /// [`Self::begin`]: the batch may have blended in part, and a failed command buffer from an earlier batch
    /// surfaces only once, at the next wait
    pub fn add(&mut self, tiles: &[u8], origins: &[[usize; 3]]) -> Result<Vec<TileStats>> {
        let out = self.add_batch(tiles, origins);
        if out.is_err() {
            self.col = None;
        }
        out
    }

    fn add_batch(&mut self, tiles: &[u8], origins: &[[usize; 3]]) -> Result<Vec<TileStats>> {
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

        // the statistics on the host while the device runs the previous batch; the uploads then wait for it
        let floor = self.recipe.std_floor;
        let stats: Vec<TileStats> = tiles.chunks_exact(vox).map(|t| TileStats::of(t, floor)).collect::<Result<_>>()?;
        let tables: Vec<f32> = stats.iter().flat_map(|s| s.table()).collect();
        self.gpu.upload_into(&self.tiles, 0, tiles)?;
        self.gpu.upload_into(&self.tables, 0, &tables)?;

        let border = self.recipe.border;
        let n: usize = shape.iter().product();
        for (i, o) in origins.iter().enumerate() {
            let z = clip(o[0], border, 0, shape[0], p);
            let y = clip(o[1], border, y0, shape[1], p);
            let x = clip(o[2], border, x0, shape[2], p);
            // the reference's `add` returns before blending a tile that misses the column; its forward still ran
            self.model.forward(&self.gpu, self.tiles.at::<u8>(i * vox), self.tables.at::<f32>(i * 256))?;
            let (Some(z), Some(y), Some(x)) = (z, y, x) else { continue };
            let extent = [z.1 - z.0, y.1 - y.0, x.1 - x.0];
            let acc = self.acc.as_ref().ok_or_else(|| eyre!("no accumulators"))?;
            let wsum = self.wsum.as_ref().ok_or_else(|| eyre!("no accumulators"))?;
            let h = self.model.head();
            let mut args = vec![acc.arg(), wsum.arg(), h.x.arg(), h.w.arg(), h.b.arg(), self.ramp.arg()];
            args.extend(
                [p, h.cin, z.0, y.0, x.0, extent[0], extent[1], extent[2], z.2, y.2, x.2, shape[1], shape[2], n]
                    .map(Arg::from),
            );
            self.gpu.dispatch_1d("head_fold", extent.iter().product(), &args)?;
        }
        // start the batch on the GPU now, so the caller's next reads and statistics overlap it
        self.gpu.flush();
        if let Some(c) = self.col.as_mut() {
            c.tiles += count;
        }
        Ok(stats)
    }

    /// Normalises the column and returns its outputs, or only its uncovered count when `quantize` is false (the
    /// reference writes nothing for a column whose tiles were all air). Fails on a non-finite probability, as the
    /// reference's `quantize` does
    pub fn finish(&mut self, quantize: bool) -> Result<ColumnOut> {
        self.finish_into(quantize, Vec::new(), Vec::new())
    }

    /// [`Self::finish`], downloading the outputs into `semantic`'s and `tracing`'s allocations, whatever they held,
    /// so a caller that hands back each finished column's two vectors allocates them about once per run instead of
    /// once per column. A vector without room for its output is replaced by one sized exactly to it, and vectors the
    /// column does not quantize into are dropped
    pub fn finish_into(&mut self, quantize: bool, mut semantic: Vec<u8>, mut tracing: Vec<u8>) -> Result<ColumnOut> {
        let col = self.col.take().ok_or_else(|| eyre!("finish before begin"))?;
        let shape = col.shape;
        let n: usize = shape.iter().product();
        let acc = self.acc.as_ref().ok_or_else(|| eyre!("no accumulators"))?;
        let wsum = self.wsum.as_ref().ok_or_else(|| eyre!("no accumulators"))?;
        let sem = ensure_buf(&self.gpu, &mut self.sem, 2 * n)?;
        let args = [sem.arg(), acc.arg(), self.stat.arg(), wsum.arg(), n.into()];
        self.gpu.dispatch("af_finish", [n.div_ceil(256), 1, 1], [256, 1, 1], &args)?;
        let half = shape.map(|v| v / 2);
        let hn: usize = half.iter().product();
        let tra = ensure_buf(&self.gpu, &mut self.tra, (2 * hn).max(1))?;
        if hn > 0 {
            let args = [tra.arg(), self.stat.arg(), acc.arg(), shape[0].into(), shape[1].into(), shape[2].into()];
            self.gpu.dispatch_1d("af_pool", 2 * hn, &args)?;
        }
        let mut status = [0u32; 2];
        self.gpu.download(&self.stat, 0, &mut status)?;
        let uncovered = u64::from(status[1]);
        if !quantize {
            return Ok(ColumnOut { shape, semantic: None, tracing: None, uncovered, tiles: col.tiles });
        }
        ensure!(status[0] & NONFINITE == 0, "non-finite probabilities");
        self.gpu.download_bytes_into(sem, 0, 2 * n, &mut semantic)?;
        self.gpu.download_bytes_into(tra, 0, 2 * hn, &mut tracing)?;
        Ok(ColumnOut { shape, semantic: Some(semantic), tracing: Some(tracing), uncovered, tiles: col.tiles })
    }

    /// Per-kernel GPU time since the last call, when `VOLKIT_METAL_PROFILE` is set, printed to stderr. It waits for
    /// the queued work, so a failure there drops the column as [`Self::add`]'s does
    pub fn print_profile(&mut self) -> Result<()> {
        let out = self.gpu.print_profile();
        if out.is_err() {
            self.col = None;
        }
        out
    }
}
