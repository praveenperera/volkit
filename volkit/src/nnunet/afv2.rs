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
//!
//! On macOS the engine is the Metal backend: the same API and pipeline, with the convolutions and GEMMs on Metal 4
//! tensor ops with true float32 operands instead of TF32, the stride-1 convolutions of levels 1 and 2 as Winograd
//! F(4,3) (`VOLKIT_METAL_WINOGRAD` chooses the levels), and the seg head folded into the blend. Its `Options` has no
//! cuDNN algorithm choice, and bf16 is CUDA-only

use color_eyre::eyre::{Result, ensure, eyre};
use serde::Serialize;

use super::Precision;

#[cfg(not(target_os = "macos"))]
mod cuda;
#[cfg(target_os = "macos")]
mod metal;

#[cfg(not(target_os = "macos"))]
pub use cuda::{AfvV2, LoadReport, Options};
#[cfg(target_os = "macos")]
pub use metal::{AfvV2, LoadReport, Options};

/// `af_finish` / `af_pool`'s status bits (volkit-kernels `afv2.rs`)
pub(crate) const NONFINITE: u32 = 1;
/// The seg head's classes: background, vertical, horizontal, intersection
pub(crate) const CLASSES: usize = 4;
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
    pub(crate) fn net(self) -> Precision {
        match self {
            Self::Fp32 => Precision::Tf32,
            Self::Bf16 => Precision::Bf16,
        }
    }
}

/// One column's outputs: `semantic` `(2, nz, ny, nx)` and `tracing` `(2, nz / 2, ny / 2, nx / 2)` uint8, channels
/// vertical then horizontal, or `None` when no tile ran (the reference writes nothing for an all-air column)
#[derive(Debug)]
#[non_exhaustive]
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
#[non_exhaustive]
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
pub(crate) struct Column {
    pub(crate) shape: [usize; 3],
    pub(crate) y0: usize,
    pub(crate) x0: usize,
    pub(crate) tiles: usize,
}

/// The tile voxels `[lo, hi)` on one axis whose volume coordinates `origin - border + i` fall in the column's
/// `[c0, c0 + len)`, and the column coordinate of the first: the reference `add`'s clip
pub(crate) fn clip(origin: usize, border: usize, c0: usize, len: usize, p: usize) -> Option<(usize, usize, usize)> {
    // tile voxel i sits at volume coordinate origin - border + i, column coordinate that minus c0
    let lo = origin as i64 - border as i64 - c0 as i64;
    let (a, b) = (lo.max(0), (lo + p as i64).min(len as i64));
    (b > a).then(|| ((a - lo) as usize, (b - lo) as usize, a as usize))
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
