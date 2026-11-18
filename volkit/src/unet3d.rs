//! The 3D ResEnc U-Net layer program shared by villa's autoconfigured 3D U-Net and nnU-Net ResEnc-L exports
//!
//! An export's sidecar holds the network's layer program: the stem, the BasicBlockD encoder stages, the
//! transposed-conv and ConvBlock decoder stages and the 1x1 head, each conv with its tensor names, channels,
//! kernel and stride. [`model::Model`] builds the network from that program, so any checkpoint inside the op set
//! the export accepts runs without code changes. The forward is channels-last at batch 1 in bf16, fp16 or float32
//! ([`Precision`]): cuDNN runs the convs, cuBLAS the transposed convs as GEMMs, and the `u3*_` kernels the
//! InstanceNorm, LeakyReLU, residuals, pooling and concat. [`zio`] reads and writes the zarr arrays around it

use color_eyre::eyre::Result;
use cudakit::{ComputeCapability, PtxTier};
use serde::Deserialize;

use crate::ensure_supported;

pub mod model;
pub mod zio;

/// The precision the network runs in: the activations' storage type, the weights' rounding, the cuDNN and
/// cuBLAS data type and the glue kernels (`u3_*` for bf16 and `u3h_*` for fp16, as autocast runs them; `u3f_*`
/// for `tf32`, float32 tensors with TF32 convolution and GEMM math, as PyTorch's float32 defaults run them)
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// FP16 storage with FP16 tensor-op math
    Fp16,
    /// BF16 storage with BF16 tensor-op math
    Bf16,
    /// Float32 storage with TF32 convolution and GEMM math
    Tf32,
}

/// A 16-bit storage type, whose values the host handles as raw bits
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Half {
    /// IEEE binary16
    Fp16,
    /// Brain floating point, 16 bits
    Bf16,
}

impl Half {
    /// `v` rounded to the storage type, as raw bits
    pub fn bits(self, v: f32) -> u16 {
        match self {
            Self::Bf16 => half::bf16::from_f32(v).to_bits(),
            Self::Fp16 => half::f16::from_f32(v).to_bits(),
        }
    }

    /// Raw storage bits as float32
    pub fn value(self, bits: u16) -> f32 {
        match self {
            Self::Bf16 => half::bf16::from_bits(bits).to_f32(),
            Self::Fp16 => half::f16::from_bits(bits).to_f32(),
        }
    }

    /// `v` as autocast casts it: rounded to the storage type, kept as float32
    pub fn round(self, v: f32) -> f32 {
        self.value(self.bits(v))
    }
}

impl Precision {
    /// The 16-bit storage type, or `None` for float32
    pub fn half(self) -> Option<Half> {
        match self {
            Self::Fp16 => Some(Half::Fp16),
            Self::Bf16 => Some(Half::Bf16),
            Self::Tf32 => None,
        }
    }

    /// Bytes per stored value
    pub fn size(self) -> usize {
        if self.half().is_some() { 2 } else { 4 }
    }

    /// `v` as the storage type holds it, kept as float32 (float32 storage keeps it unchanged)
    pub fn round(self, v: f32) -> f32 {
        self.half().map_or(v, |h| h.round(v))
    }

    /// `values` rounded to the storage type, as the little-endian bytes the GPU holds
    pub fn encode(self, values: &[f32]) -> Vec<u8> {
        match self.half() {
            Some(h) => values.iter().flat_map(|&v| h.bits(v).to_le_bytes()).collect(),
            None => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    /// Stored little-endian bytes as float32, the inverse of [`Precision::encode`]
    pub fn decode(self, bytes: &[u8]) -> Vec<f32> {
        match self.half() {
            Some(h) => bytes.as_chunks::<2>().0.iter().map(|&b| h.value(u16::from_le_bytes(b))).collect(),
            None => bytes.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect(),
        }
    }
}

/// An InstanceNorm's affine tensors and eps
#[derive(Clone, Debug, Deserialize)]
pub struct NormSpec {
    /// Tensor name of the scale
    pub w: String,
    /// Tensor name of the shift
    pub b: String,
    /// Added to the variance for numerical stability
    pub eps: f32,
}

/// One conv, with the norm and activation that follow it when it is a ConvDropoutNormReLU
#[derive(Clone, Debug, Deserialize)]
pub struct ConvSpec {
    /// Tensor name of the weight
    pub w: String,
    /// Tensor name of the bias, if the conv has one
    pub b: Option<String>,
    /// Input channels
    pub cin: usize,
    /// Output channels
    pub cout: usize,
    /// Kernel size per axis, `[z, y, x]`
    pub kernel: Vec<usize>,
    /// Stride per axis, `[z, y, x]`
    pub stride: Vec<usize>,
    /// The norm that follows the conv, if any
    pub norm: Option<NormSpec>,
    /// LeakyReLU's negative slope, or `None` for no activation
    pub leaky: Option<f32>,
}

/// A BasicBlockD skip: optional average pool (kernel = stride), then optional bias-free 1x1 conv + norm
#[derive(Clone, Debug, Deserialize)]
pub struct SkipSpec {
    /// The average pool applied to the skip path
    pub pool: Pool,
    /// The 1x1 conv and norm after the pool, if any
    pub proj: Option<ConvSpec>,
}

/// A skip's average-pool stride: one value for every axis (`inkrs-unet3d-v1`) or one per axis
/// (`inkrs-nnunet-v1`); 1 means no pooling on that axis
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum Pool {
    /// The same stride on every axis
    Iso(usize),
    /// A stride per axis, `[z, y, x]`
    Axes([usize; 3]),
}

impl Pool {
    /// The stride per axis, z, y, x
    pub fn axes(self) -> [usize; 3] {
        match self {
            Self::Iso(v) => [v; 3],
            Self::Axes(a) => a,
        }
    }
}

/// One BasicBlockD: `leaky(norm(conv2(leaky(norm(conv1(x))))) + skip(x))`
#[derive(Clone, Debug, Deserialize)]
pub struct BlockSpec {
    /// The first conv of the block
    pub conv1: ConvSpec,
    /// The second conv, whose output the skip is added to
    pub conv2: ConvSpec,
    /// The skip path
    pub skip: SkipSpec,
    /// LeakyReLU's negative slope after the add
    pub leaky: f32,
}

/// One decoder stage: k2 s2 transposed conv, concat `(up, skip)`, then its convs
#[derive(Clone, Debug, Deserialize)]
pub struct DecSpec {
    /// The k2 s2 transposed conv
    pub transp: ConvSpec,
    /// The convs that follow the concat
    pub convs: Vec<ConvSpec>,
}

/// The whole layer program
#[derive(Clone, Debug, Deserialize)]
pub struct Program {
    /// The stem's convs
    pub stem: Vec<ConvSpec>,
    /// The encoder stages, each a list of blocks
    pub stages: Vec<Vec<BlockSpec>>,
    /// The decoder stages, deepest first
    pub decoder: Vec<DecSpec>,
    /// The segmentation head, a 1x1 conv
    pub head: ConvSpec,
}

/// Fails with [`Unsupported`](crate::error::Unsupported) on a device the engine can't run on: its convs are cuDNN bf16, which
/// needs compute capability 8.0, and an sm75 PTX tier, which volkit does not ship, would lack its kernels
pub fn check_device(cc: ComputeCapability, tier: PtxTier) -> Result<()> {
    ensure_supported!(
        cc >= ComputeCapability::new(8, 0) && tier != PtxTier::Sm75,
        "the 3D U-Net engines (ink3d, nnunet, fiber, afv-v2) need a GPU of compute capability 8.0 or above and the \
         sm80 or sm120 PTX tier, for their cuDNN bf16, fp16 and TF32 convs; this GPU is {cc} on the {tier} tier"
    );
    Ok(())
}

impl ConvSpec {
    /// The kernel and stride as three axes, each kernel odd "same" and each stride 1 or 2 when `odd`
    fn check(&self, what: &str, cin: usize, odd: bool) -> Result<()> {
        ensure_supported!(self.cin == cin, "{what}: {} input channels, the previous layer gives {cin}", self.cin);
        ensure_supported!(self.kernel.len() == 3 && self.stride.len() == 3, "{what}: not 3D");
        if odd {
            ensure_supported!(
                self.kernel.iter().all(|&k| k % 2 == 1) && self.stride.iter().all(|&s| s == 1 || s == 2),
                "{what}: kernel {:?} stride {:?}",
                self.kernel,
                self.stride
            );
        }
        ensure_supported!(
            self.cout.is_multiple_of(2) || self.norm.is_none(),
            "{what}: odd channel count {}",
            self.cout
        );
        if let Some(n) = &self.norm {
            ensure_supported!(n.eps > 0.0 && n.eps.is_finite(), "{what}: eps {}", n.eps);
        }
        Ok(())
    }

    fn axes(v: &[usize]) -> [usize; 3] {
        [v[0], v[1], v[2]]
    }

    /// Whether kernel and stride are the same on every axis
    fn isotropic(&self) -> bool {
        self.kernel.iter().all(|&v| v == self.kernel[0]) && self.stride.iter().all(|&v| v == self.stride[0])
    }

    /// Multiply-accumulates on an input of `shape`, and the output shape ("same" padding)
    fn macs(&self, shape: [usize; 3]) -> (u64, [usize; 3]) {
        let out: [usize; 3] = std::array::from_fn(|a| shape[a].div_ceil(self.stride[a]));
        let taps: usize = self.kernel.iter().product();
        let vox: usize = out.iter().product();
        ((vox * self.cout * self.cin * taps) as u64, out)
    }
}

impl Program {
    /// Every layer's channels chain from `in_channels`, each stage's stride divides its input evenly, the
    /// decoder mirrors the encoder (a transposed conv of kernel = stride undoing each stage's stride, with
    /// bias) and the head is a 1x1 conv with bias. Returns each stage's output shape and channels, for a
    /// patch of `patch` voxels (z, y, x)
    pub fn levels(&self, patch: [usize; 3], in_channels: usize) -> Result<Vec<([usize; 3], usize)>> {
        ensure_supported!(!self.stages.is_empty() && self.stages.iter().all(|s| !s.is_empty()), "empty stage");
        ensure_supported!(patch.iter().all(|&v| v > 0 && v < 1 << 10), "patch {patch:?}");
        let mut cin = in_channels;
        for (i, s) in self.stem.iter().enumerate() {
            s.check(&format!("stem {i}"), cin, true)?;
            ensure_supported!(
                s.stride.iter().all(|&v| v == 1) && s.norm.is_some() && s.leaky.is_some(),
                "stem {i}: stride or activation"
            );
            cin = s.cout;
        }
        let mut shape = patch;
        let mut levels = Vec::new();
        let mut strides = Vec::new();
        for (si, stage) in self.stages.iter().enumerate() {
            let stride = ConvSpec::axes(&stage[0].conv1.stride);
            for (bi, b) in stage.iter().enumerate() {
                let what = format!("stage {si} block {bi}");
                b.conv1.check(&format!("{what} conv1"), cin, true)?;
                b.conv2.check(&format!("{what} conv2"), b.conv1.cout, true)?;
                ensure_supported!(b.conv2.stride.iter().all(|&v| v == 1), "{what}: conv2 stride {:?}", b.conv2.stride);
                ensure_supported!(b.conv1.norm.is_some() && b.conv1.leaky.is_some(), "{what}: conv1 norm/act");
                ensure_supported!(b.conv2.norm.is_some() && b.conv2.leaky.is_none(), "{what}: conv2 norm/act");
                let s = ConvSpec::axes(&b.conv1.stride);
                ensure_supported!(b.skip.pool.axes() == s, "{what}: skip pool {:?} with stride {s:?}", b.skip.pool);
                ensure_supported!(bi == 0 || s == [1; 3], "{what}: stride {s:?} past a stage's first block");
                for a in 0..3 {
                    ensure_supported!(
                        shape[a].is_multiple_of(s[a]),
                        "{what}: shape {shape:?} does not divide by {s:?}"
                    );
                    shape[a] /= s[a];
                }
                match &b.skip.proj {
                    Some(pj) => {
                        pj.check(&format!("{what} skip"), cin, true)?;
                        ensure_supported!(
                            pj.kernel.iter().all(|&v| v == 1)
                                && pj.stride.iter().all(|&v| v == 1)
                                && pj.b.is_none()
                                && pj.norm.is_some()
                                && pj.leaky.is_none(),
                            "{what}: skip projection must be a bias-free 1x1 conv + norm"
                        );
                        ensure_supported!(pj.cout == b.conv2.cout, "{what}: skip gives {} channels", pj.cout);
                    }
                    None => {
                        ensure_supported!(cin == b.conv2.cout, "{what}: identity skip across {cin} -> {}", b.conv2.cout)
                    }
                }
                cin = b.conv2.cout;
            }
            levels.push((shape, cin));
            strides.push(stride);
        }
        ensure_supported!(self.decoder.len() + 1 == self.stages.len(), "{} decoder stages", self.decoder.len());
        for (d, dec) in self.decoder.iter().enumerate() {
            let level = self.stages.len() - 2 - d;
            let (_, skip_c) = levels[level];
            let t = &dec.transp;
            t.check(&format!("decoder {d} transp"), cin, false)?;
            let undo = strides[level + 1];
            ensure_supported!(
                ConvSpec::axes(&t.kernel) == undo && ConvSpec::axes(&t.stride) == undo && t.b.is_some(),
                "decoder {d}: transp k{:?} s{:?} must undo stride {undo:?}, with bias",
                t.kernel,
                t.stride
            );
            ensure_supported!(
                t.cout == skip_c && t.cout % 2 == 0,
                "decoder {d}: transp to {} channels, skip has {skip_c}",
                t.cout
            );
            ensure_supported!(!dec.convs.is_empty(), "decoder {d}: no convs");
            let mut c = 2 * skip_c;
            for (i, cv) in dec.convs.iter().enumerate() {
                cv.check(&format!("decoder {d} conv {i}"), c, true)?;
                ensure_supported!(
                    cv.stride.iter().all(|&v| v == 1) && cv.norm.is_some() && cv.leaky.is_some(),
                    "decoder {d} conv {i}"
                );
                c = cv.cout;
            }
            cin = c;
        }
        let h = &self.head;
        h.check("head", cin, true)?;
        ensure_supported!(
            h.kernel.iter().all(|&v| v == 1) && h.cout >= 1 && h.b.is_some() && h.norm.is_none() && h.leaky.is_none(),
            "head must be a 1x1 conv with bias"
        );
        Ok(levels)
    }

    /// The layers [`model::Model`] runs: cubic kernels, isotropic strides and pools, k2 s2 transposed convs
    pub fn check_isotropic(&self) -> Result<()> {
        let convs = self
            .stem
            .iter()
            .chain(self.stages.iter().flatten().flat_map(|b| [&b.conv1, &b.conv2].into_iter().chain(&b.skip.proj)))
            .chain(self.decoder.iter().flat_map(|d| d.convs.iter()))
            .chain([&self.head]);
        for c in convs {
            ensure_supported!(c.isotropic(), "{}: kernel {:?} stride {:?}, only isotropic", c.w, c.kernel, c.stride);
        }
        for d in &self.decoder {
            let t = &d.transp;
            ensure_supported!(
                t.kernel.iter().all(|&v| v == 2) && t.stride.iter().all(|&v| v == 2),
                "{}: transp k{:?} s{:?}, only k2 s2",
                t.w,
                t.kernel,
                t.stride
            );
        }
        for b in self.stages.iter().flatten() {
            let pool = b.skip.pool.axes();
            ensure_supported!(pool.iter().all(|&v| v == pool[0]), "skip pool {pool:?}, only isotropic");
        }
        Ok(())
    }

    /// The layers [`model::Model`] runs: cubic kernels, a stride of 1 or 2 on each axis (AFV's last stage
    /// strides (2, 2, 1)), skip pools equal to their stride, and transposed convs whose kernel equals their
    /// stride on each axis. [`Program::check_isotropic`] is the narrower contract v0.2's models and m7 keep
    pub fn check_axes(&self) -> Result<()> {
        let convs = self
            .stem
            .iter()
            .chain(self.stages.iter().flatten().flat_map(|b| [&b.conv1, &b.conv2].into_iter().chain(&b.skip.proj)))
            .chain(self.decoder.iter().flat_map(|d| d.convs.iter()))
            .chain([&self.head]);
        for c in convs {
            ensure_supported!(
                c.kernel.len() == 3 && c.kernel.iter().all(|&v| v == c.kernel[0]),
                "{}: kernel {:?}, only cubic",
                c.w,
                c.kernel
            );
            ensure_supported!(
                c.stride.len() == 3 && c.stride.iter().all(|&v| v == 1 || v == 2),
                "{}: stride {:?}, 1 or 2 per axis",
                c.w,
                c.stride
            );
        }
        for d in &self.decoder {
            let t = &d.transp;
            ensure_supported!(
                t.kernel == t.stride && t.stride.len() == 3 && t.stride.iter().all(|&v| v == 1 || v == 2),
                "{}: transp k{:?} s{:?}, kernel = stride of 1 or 2 per axis",
                t.w,
                t.kernel,
                t.stride
            );
        }
        for b in self.stages.iter().flatten() {
            let pool = b.skip.pool.axes();
            ensure_supported!(pool.iter().all(|&v| v == 1 || v == 2), "skip pool {pool:?}, 1 or 2 per axis");
        }
        Ok(())
    }

    /// Multiply-accumulates of one forward on a `patch` input: every conv (output voxels x Cout x Cin x
    /// taps), every transposed conv (input voxels x Cin x Cout x taps) and the head
    pub fn macs(&self, patch: [usize; 3]) -> u64 {
        let mut shape = patch;
        let mut total = 0u64;
        let conv = |total: &mut u64, c: &ConvSpec, s: [usize; 3]| -> [usize; 3] {
            let (m, out) = c.macs(s);
            *total += m;
            out
        };
        for s in &self.stem {
            shape = conv(&mut total, s, shape);
        }
        let mut shapes = Vec::new();
        for stage in &self.stages {
            for b in stage {
                if let Some(pj) = &b.skip.proj {
                    let pool = b.skip.pool.axes();
                    conv(&mut total, pj, std::array::from_fn(|a| shape[a] / pool[a]));
                }
                shape = conv(&mut total, &b.conv1, shape);
                shape = conv(&mut total, &b.conv2, shape);
            }
            shapes.push(shape);
        }
        for (d, dec) in self.decoder.iter().enumerate() {
            let t = &dec.transp;
            let taps: usize = t.kernel.iter().product();
            total += (shape.iter().product::<usize>() * t.cin * t.cout * taps) as u64;
            shape = shapes[shapes.len() - 2 - d];
            for c in &dec.convs {
                shape = conv(&mut total, c, shape);
            }
        }
        conv(&mut total, &self.head, shape);
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ink3d is refused as unsupported below compute capability 8.0 and on the sm75 tier, forced or
    /// not, so callers fall back instead of failing at a missing kernel or a cuDNN bf16 plan
    #[test]
    fn ink3d_needs_sm80() {
        let cc = ComputeCapability::new;
        for (c, tier) in [(cc(7, 5), PtxTier::Sm75), (cc(12, 0), PtxTier::Sm75)] {
            let err = check_device(c, tier).unwrap_err();
            assert!(crate::error::is_unsupported(&err), "{err}");
        }
        assert!(check_device(cc(8, 6), PtxTier::Sm80).is_ok());
        assert!(check_device(cc(12, 0), PtxTier::Sm80).is_ok());
        assert!(check_device(cc(12, 0), PtxTier::Sm120).is_ok());
    }
}
