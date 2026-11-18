# volkit API

What a consumer such as fibrs calls to run AFV v2 (`afv_fiber_9um_v2`) natively. The names and signatures are
inkrs's (`inkrs::nnunet::afv2` at inkrs `c1a9850`, the code behind `inkrs.AfvV2`), moved under `volkit::`, so a
consumer switches by import path. Items marked *new* have no inkrs counterpart.

The crate builds the CUDA runtime on Linux and Windows. On macOS it builds empty, so gate the dependency:

```toml
[target.'cfg(target_os = "linux")'.dependencies]
volkit = "0.1"
```

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
    /// The reference contract: float32 tensors with TF32 conv and GEMM math, as PyTorch's float32 defaults
    #[default]
    Fp32,
    /// Opt-in, changed output: PyTorch's bf16 autocast forward
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
pub struct LoadReport {
    pub model: String,
    pub precision: AfvPrecision,
    pub patch: usize,
    pub border: usize,
    pub load_s: f64,
    pub plans: Vec<String>,
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
}
```

The caller keeps the reference's (`lasagna/qualzz_inference.py`) tiling, reads and air skipping: 128^3 tiles on a
lattice of stride 96 over the band padded by 16, grouped into columns. Per column: `begin`, then `add` with the
column's non-air tiles in the reference's order, then `finish`. This is exactly the sequence `inkrs.AfvV2`'s Python
layer drives; volkit has no tiler for it (`nnunet::grid` and `nnunet::engine` are the `vesuvius.predict` engine,
which AFV v2 does not use, and stay in inkrs).

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

## Errors

Functions return `color_eyre::eyre::Result`. `volkit::error::is_unsupported(&report)` is true when the model or
contract is one volkit does not implement (a fallback case rather than a fault), and
`volkit::error::is_library_error(&report)` when a CUDA library could not be loaded or trusted.
