# volkit API

What a consumer such as fibrs calls to run AFV v2 (`afv_fiber_9um_v2`) natively. The names and signatures are
inkrs's (`inkrs::nnunet::afv2` at inkrs `c1a9850`, the code behind `inkrs.AfvV2`), moved under `volkit::`, so a
consumer switches by import path. Items marked *new* have no inkrs counterpart.

The crate builds the CUDA runtime on Linux and Windows and the Metal runtime on macOS (Apple silicon). Both expose
the `AfvV2` API below; only `Options` and `LoadReport` differ, as [Metal](#metal) lists.

```toml
[dependencies]
volkit = "0.2"
```

volkit defaults to parity with the reference runner: `AfvPrecision::Fp32` is the default, and `AfvPrecision::Bf16` is an
opt-in speed mode with small, measured differences (the
[README](https://github.com/praveenperera/volkit#precision) gives the figures). Every speed figure names the
precision it was measured at.

At run time it needs an NVIDIA driver, `libcudnn.so.9` (cuDNN 9) and `libcublas.so.12` on the loader path. cudakit
loads them with `dlopen` and checks their symbols before the first call, so a missing library is an error from
`AfvV2::open`, not a crash.

## `volkit::nnunet::afv2`

```rust
use volkit::dnn::Pick;
use volkit::nnunet::afv2::{AfvPrecision, AfvV2, ColumnOut, LoadReport, MAX_BATCH, Options, TileStats};
```

```rust
/// Tiles one `AfvV2::add` call takes at most (the reference batches 4)
pub const MAX_BATCH: usize = 16;

/// The arithmetic the network runs in
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AfvPrecision {
    /// The default, for parity with the reference runner (villa's `lasagna.qualzz_inference` on PyTorch): float32
    /// tensors with TF32 tensor-core conv and GEMM math, as PyTorch's float32 defaults run the model
    #[default]
    Fp32,
    /// Opt-in speed mode with small, measured differences: PyTorch's bf16 autocast forward. Against float32,
    /// about 99% of traced fibre length agrees (99.2-99.4%) and 97.0-97.8% of fibres match. Never the default
    Bf16,
}
impl AfvPrecision {
    /// `"fp32"` or `"bf16"`
    pub fn parse(name: &str) -> Result<Self>;
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// CUDA device ordinal
    pub device: usize,
    /// `Pick::Find` (time every cuDNN algorithm, what `inkrs.AfvV2(algo="find")` defaults to) or `Pick::Heuristic`
    pub pick: Pick,
    pub precision: AfvPrecision,
}

/// What loading the engine took and chose (serialisable to JSON)
#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct LoadReport {
    pub model: String,
    pub precision: AfvPrecision,
    pub patch: usize,
    pub border: usize,
    pub load_s: f64,
    /// Each conv plan with the time cuDNN measured for it, `shape | ALGO MATH ws N B, T ms`
    pub plans: Vec<String>,
    /// (new) The algorithm choices without timings, `shape | ALGO MATH ws N B`: each distinct conv in forward
    /// order, then the float32 seg head (none for the bf16 head). Two loads chose the same cuDNN algorithms
    /// exactly when these lists are equal. That covers the cuDNN choice only: compare `cudnn` and `cublas` too,
    /// since another cuDNN or cuBLAS release can round the same choices differently (the decoder's GEMMs run on
    /// cuBLAS)
    pub algos: Vec<String>,
    pub head_plan: String,
    pub head_channels: usize,
    pub algo_cache: volkit::dnn::algo_cache::Status,
    pub cudnn: usize,
    /// (new) `cublasGetVersion` of the cuBLAS loaded, such as 120804; see the README on library versions
    pub cublas: i32,
    pub stem_padded: bool,
    pub ptx_tier: String,
    pub gpu: String,
    pub flops_per_tile: u64,
}

/// One column's outputs, uint8, channels vertical then horizontal: `semantic` `(2, nz, ny, nx)` and `tracing`
/// `(2, nz / 2, ny / 2, nx / 2)`, C order; `None` when `finish(false)` (the reference writes nothing for an
/// all-air column)
#[derive(Debug)]
#[non_exhaustive]
pub struct ColumnOut {
    pub shape: [usize; 3],
    pub semantic: Option<Vec<u8>>,
    pub tracing: Option<Vec<u8>>,
    /// Voxels no tile weighted (the reference's `voxels_uncovered`)
    pub uncovered: u64,
    pub tiles: usize,
}

/// A tile's normalisation: float32 mean and floored unbiased std
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[non_exhaustive]
pub struct TileStats {
    pub mean: f32,
    pub std: f32,
}
impl TileStats {
    pub fn of(tile: &[u8], floor: f32) -> Result<Self>;
    pub fn table(&self) -> [f32; 256];
}

/// An AFV v2 export on the GPU, accumulating one output column at a time. `Send`, not `Sync`: wrap it in a
/// `Mutex` to share it
pub struct AfvV2 {
    pub load: LoadReport,
    /* private */
}
impl AfvV2 {
    /// Loads the model at `model` and plans the forward and the seg head. `model` is either the `.safetensors`
    /// export of `export_nnunet.py --recipe afv-v2` with its `.json` sidecar beside it, or (new) the nnU-Net
    /// model folder itself (`plans.json`, `dataset.json`, `checkpoint_final.pth`), converted in memory without
    /// torch by `volkit::nnunet::export::Export::afv_v2`; nothing is written
    pub fn open(model: &Path, opts: &Options) -> Result<Self>;

    pub fn precision(&self) -> AfvPrecision;
    /// The tile edge (128)
    pub fn patch(&self) -> usize;
    /// The tile border the blend ramp zeroes (16)
    pub fn border(&self) -> usize;
    /// The normalisation's floor on the std (10.0)
    pub fn std_floor(&self) -> f32;

    /// Starts the column of `shape` `(nz, ny, nx)` whose first voxel is band voxel `(0, y0, x0)`, with zeroed
    /// accumulators; a column begun and not finished is dropped
    pub fn begin(&mut self, shape: [usize; 3], y0: usize, x0: usize) -> Result<()>;

    /// Runs `origins.len()` tiles (at most `MAX_BATCH`) and blends them into the column, in order. `tiles` holds
    /// their `p^3` uint8 inputs back to back, C order `(z, y, x)`: the band read at `origin - border` and
    /// reflected at the band's edges, as the reference's `read_tile` does. `origins` are their padded
    /// `(pz, py, px)` lattice origins. Returns each tile's normalisation
    pub fn add(&mut self, tiles: &[u8], origins: &[[usize; 3]]) -> Result<Vec<TileStats>>;

    /// Normalises the column by its weights and returns its outputs, or only its uncovered count when
    /// `quantize` is false. Fails on a non-finite probability, as the reference's `quantize` does
    pub fn finish(&mut self, quantize: bool) -> Result<ColumnOut>;
    /// (new) `finish`, downloading the outputs into `semantic`'s and `tracing`'s allocations, whatever they held,
    /// so a caller that hands back each finished column's two vectors allocates them about once per run instead of
    /// once per column (about 520 MB a column at nz 1000). A vector without room for its output is replaced by one
    /// sized exactly to it, and vectors the column does not quantize into are dropped
    pub fn finish_into(&mut self, quantize: bool, semantic: Vec<u8>, tracing: Vec<u8>) -> Result<ColumnOut>;
}
```

The caller keeps the reference's (`lasagna/qualzz_inference.py`) tiling, reads and air skipping: 128^3 tiles on a
lattice of stride 96 over the band padded by 16, grouped into columns. Per column: `begin`, then `add` with the
column's non-air tiles in the reference's order, then `finish` (or `finish_into` with the last column's vectors, once
the caller has written them). This is exactly the sequence `inkrs.AfvV2`'s Python
layer drives; volkit has no tiler for it (`nnunet::grid` and `nnunet::engine` are the `vesuvius.predict` engine,
which AFV v2 does not use, and stay in inkrs).

### Metal

On macOS, `volkit::nnunet::afv2` is the Metal engine: the same `AfvV2`, `AfvPrecision`, `ColumnOut`, `TileStats` and
`MAX_BATCH`, with these `Options` and `LoadReport`. It needs macOS 26 or newer (Metal 4) and Apple silicon (M1 or
newer); `AfvV2::open` fails with an unsupported error (`volkit::error::is_unsupported`) on anything else, on a device
other than 0, and for `AfvPrecision::Bf16`, which is CUDA-only. The kernels are MSL compiled at `open`, so a build
needs neither Xcode nor the Metal toolchain.

```rust
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Metal device ordinal: Apple silicon has one GPU, device 0
    pub device: usize,
    /// Only `AfvPrecision::Fp32` runs on Metal
    pub precision: AfvPrecision,
}

#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct LoadReport {
    pub model: String,
    pub precision: AfvPrecision,
    pub patch: usize,
    pub border: usize,
    /// Seconds spent loading the weights and compiling the kernels
    pub load_s: f64,
    /// The device's name, such as "Apple M4 Pro"
    pub gpu: String,
    /// Bytes the model's weights and activation buffers hold on the device
    pub model_bytes: usize,
    /// The device's recommended working set (`recommendedMaxWorkingSetSize`): the most this process should keep
    /// allocated on the GPU before Metal starts paging
    pub working_set: usize,
    pub flops_per_tile: u64,
    /// The levels whose stride-1 convolutions run Winograd F(4,3), ascending; outputs match byte for byte only
    /// between equal lists
    pub winograd_levels: Vec<usize>,
}

impl AfvV2 {
    /// Prints each kernel's GPU time, when `VOLKIT_METAL_PROFILE` is set
    pub fn print_profile(&self) -> Result<()>;
}
```

`Fp32` on Metal runs the convolutions and GEMMs with true float32 operands where CUDA runs TF32, so Metal's output sits
inside TF32's rounding noise of CUDA's, not byte-identical to it. The stride-1 convolutions of levels 0 to 2 run as
Winograd F(4,3), whose rounding differs from a direct convolution's by a few float32 ulps; `VOLKIT_METAL_WINOGRAD`
chooses the levels, and a level runs direct convolutions where its Winograd scratch (a buffer of another level)
cannot hold one tile, which happens only on tiny networks. A given build, device and setting is deterministic: the
same input gives the same bytes.

`LoadReport`, `ColumnOut` and `TileStats` are `#[non_exhaustive]` (new): read their fields freely, but outside volkit
they cannot be built with a struct literal or destructured without `..`, so a later release can add fields without
a breaking change. `TileStats::of` builds one.

## `volkit::dnn`

```rust
/// How a cuDNN plan chooses its algorithm
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// cuDNN's `_v7` heuristic ranking, first supported entry
    Heuristic,
    /// `cudnnFindConvolutionForwardAlgorithmEx`: time every algorithm, take the fastest
    Find,
}
```

`Pick::Find` results are cached on disk (`volkit::dnn::algo_cache`), keyed on the GPU, driver, cuDNN and volkit
version, under `$VOLKIT_CACHE_DIR`, else `$XDG_CACHE_HOME/volkit`, else `~/.cache/volkit`. `VOLKIT_ALGO_CACHE=0`
(or `off`, `false`) turns the cache off. Same behaviour as inkrs, under volkit's names.

## `volkit::nnunet::export` (new): torch-free weights

```rust
/// An nnU-Net model folder converted to the safetensors export `export_nnunet.py` writes: the tensors by name
/// and the JSON sidecar (the inference contract)
pub struct Export {
    pub sidecar: serde_json::Value,
    /* private: the tensors, float32, and their aliases */
}
impl Export {
    /// Reads `dir` (`plans.json`, `dataset.json`, `checkpoint_final.pth`) without torch or Python and maps it as
    /// `export_nnunet.py --recipe afv-v2` does: the seg network only, the centre, affinity and direction heads
    /// dropped. The `.pth` is read with a zip reader and a restricted pickle machine that only builds tensors,
    /// dicts and inert stub objects; it never imports or calls anything the file names
    pub fn afv_v2(dir: &Path) -> Result<Self>;

    /// Writes `path` (`.safetensors`) and its `.json` sidecar beside it. The `.safetensors` is byte for byte what
    /// the Python export writes, so `AfvV2::open(path, ..)` can load it later without the conversion
    pub fn write(&self, path: &Path) -> Result<()>;

    /// The unique tensors (name -> (shape, float32 data)) and aliases (alias -> target), for an in-memory load
    pub fn into_parts(self) -> (volkit::weights::Tensors, BTreeMap<String, String>, serde_json::Value);
}
```

The reader underneath is `volkit::pth`: `Pth::open(path)` maps a `.pth`, checks its zip directory and unpickles
`data.pkl` into a `pth::Value` tree whose tensors (`pth::TensorRef`) are read with `Pth::tensor_f32`. It only
accepts little-endian float32 storages, contiguous tensors and stored (uncompressed) zip entries, the form
`torch.save` writes.

## Environment variables

| Variable | Effect |
| --- | --- |
| `VOLKIT_PTX_TIER` | Force a PTX tier at or below the device's (`sm80` or `sm120`) |
| `VOLKIT_CACHE_DIR` | Root of the on-disk caches (cuDNN algorithm choices, the driver's compiled-PTX cache) |
| `VOLKIT_ALGO_CACHE` | `0`, `off` or `false` turns the cuDNN algorithm cache off |
| `VOLKIT_METAL_WINOGRAD` | Metal: the levels whose stride-1 convolutions run Winograd F(4,3), comma-separated (default `0,1,2`), or `none` for direct convolutions everywhere |
| `VOLKIT_METAL_WINO_TILES` | Metal: the most tiles one level-0 Winograd chunk takes (default 256, small enough to stay in the GPU's cache), or `0` for as many as the scratch holds |
| `VOLKIT_METAL_PROFILE` | Metal: time every kernel with GPU counters, for `AfvV2::print_profile` (slows the run) |
| `VOLKIT_METAL_TESTS` | `1` runs the Metal kernel tests, which touch the GPU; they skip otherwise |

## Errors

Functions return `color_eyre::eyre::Result`. `volkit::error::is_unsupported(&report)` is true when the model or
contract is one volkit does not implement (a fallback case rather than a fault), and
`volkit::error::is_library_error(&report)` when a CUDA library could not be loaded or trusted. On Metal,
`is_unsupported` is also true for an OS older than macOS 26, a GPU older than Apple7 (M1), a device other than 0,
the bf16 precision, and a network the Metal forward can't run: a level width that is not a multiple of 4, a first
encoder stage that strides or a later one that does not, or a level too large for 32-bit indexing. `AfvV2::open`
checks the network before it opens the device.
