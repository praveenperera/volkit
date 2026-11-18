//! nnU-Net ResEnc-L exports: the sidecar contract and AFV v2's seg inference ([`afv2`])
//!
//! inkrs's `export_nnunet.py`, or [`export`] without torch, exports a checkpoint's weights, its layer program (the
//! same program [`crate::unet3d`] builds its forward from, with per-axis kernels, strides and pools) and the
//! recipe of the pipeline the weights run in: precision, input normalisation, test-time augmentation, blend and
//! finalisation. [`Contract`] reads and checks that sidecar on the CPU, for every recipe the format has;
//! [`afv2`] runs the `afv-v2` recipe (villa `lasagna/qualzz_inference.py`, AFV v2's seg head). The other recipes
//! (`vesuvius-predict`, `fiber-tool`, `hercunet`) parse but have no engine here

use std::collections::{BTreeMap, BTreeSet};

use color_eyre::eyre::Result;
use serde::Deserialize;

use crate::ensure_supported;
use crate::error::Unsupported;
use crate::unet3d::{ConvSpec, Program};
pub use crate::unet3d::{Half, Precision};

pub mod afv2;
pub mod export;

/// The sidecar format this module reads, as inkrs's exporter names it
pub const FORMAT: &str = "inkrs-nnunet-v1";

/// nnU-Net's CTNormalization as villa computes it: `(clip(x, low, high) - center) / scale` in float64,
/// stored as float32, one value per uint8 code
#[derive(Clone, Debug, Deserialize)]
pub struct CtNorm {
    /// Centre subtracted after clipping
    pub center: f64,
    /// Divisor applied after centring
    pub scale: f64,
    /// Lower clip bound, applied before centring
    pub low: f64,
    /// Upper clip bound, applied before centring
    pub high: f64,
    /// The 256-entry lookup table, one float32 bit pattern per uint8 code
    pub table_f32_bits: Vec<u32>,
}

impl CtNorm {
    /// The 256-entry table as float32
    pub fn table(&self) -> Vec<f32> {
        self.table_f32_bits.iter().map(|&b| f32::from_bits(b)).collect()
    }

    /// Fails unless the table is the float32 rounding of the float64 formula for every code, so the
    /// recorded parameters and the table cannot disagree
    fn check(&self) -> Result<()> {
        ensure_supported!(self.table_f32_bits.len() == 256, "CT table of {} entries", self.table_f32_bits.len());
        ensure_supported!(
            self.scale > 0.0 && self.low <= self.high,
            "CT scale {} range {}..{}",
            self.scale,
            self.low,
            self.high
        );
        for (code, &bits) in self.table_f32_bits.iter().enumerate() {
            let want = ((code as f64).clamp(self.low, self.high) - self.center) / self.scale;
            ensure_supported!(
                (want as f32).to_bits() == bits,
                "CT table entry {code}: {} for {want}",
                f32::from_bits(bits)
            );
        }
        Ok(())
    }
}

/// How the per-voxel input is normalised
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Normalization {
    /// CT normalisation through a lookup table, as the other recipes' CTNormalization
    Ct(CtNorm),
    /// Fiber tool: z-score over a whole block, std floored
    BlockZscore {
        /// Floor on the block's standard deviation
        std_floor: f64,
        /// Block size per axis, `[z, y, x]` in voxels
        block: [usize; 3],
        /// Margin around the block, in voxels
        margin: usize,
    },
    /// AFV v2's `normalize_ct`: z-score over each tile, with its unbiased std floored
    TileZscore {
        /// Floor on the tile's unbiased standard deviation
        std_floor: f64,
    },
}

/// Test-time augmentation
#[derive(Clone, Debug, Deserialize)]
pub struct Tta {
    /// `mirror8` (every flip of z, y, x, as one batch) or `none`
    pub mode: String,
    /// `mean_logits`: the flips' fp16 logits averaged
    pub combine: String,
}

/// The patch blend
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Blend {
    /// villa's `generate_gaussian_map`, stored as the float32 tensor `tensor`
    GaussianMap {
        /// Name of the exported float32 tensor holding the map
        tensor: String,
        /// The patch size divided by this gives the Gaussian's sigma
        sigma_scale: f64,
    },
    /// The fiber tool's separable Gaussian
    AnalyticGaussian {
        /// The patch size divided by this gives the Gaussian's sigma
        sigma_scale: f64,
        /// Lowest weight the Gaussian is clamped to
        floor: f64,
        /// One axis of the separable Gaussian, as float32 bit patterns
        axis_f32_bits: Vec<u32>,
    },
    /// lasagna's `blend_ramp`: per axis 0 on `border` voxels at each end and 1 on the core, with linear edges
    /// `overlap - 2 * border` long inside the core, the tile weight the product of the three axes
    Ramp {
        /// Voxels at each end of an axis whose weight is 0
        border: usize,
        /// Voxels shared by neighbouring tiles
        overlap: usize,
    },
}

/// The output finalisation
#[derive(Clone, Debug, Deserialize)]
pub struct Finalize {
    /// `softmax_u8`: villa's `finalize_outputs` (binary: `softmax[1] x 255` truncated, or a logit cutoff)
    pub mode: String,
}

/// The pipeline the weights run in
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "pipeline", rename_all = "kebab-case")]
pub enum Recipe {
    /// villa `vesuvius.predict` with `--blend_in_memory` (m7)
    VesuviusPredict {
        /// The precision the network runs in
        precision: Precision,
        /// How the input is normalised
        normalization: Normalization,
        /// Test-time augmentation
        tta: Tta,
        /// How patch predictions are blended
        blend: Blend,
        /// How the blended output is finalised
        finalize: Finalize,
        /// Patch overlap as a fraction of the patch
        overlap: f64,
        /// Whether patches whose input is constant are skipped
        skip_constant_patches: bool,
    },
    /// villa's fiber spline generator (AFV fiber)
    FiberTool {
        /// The precision the network runs in
        precision: Precision,
        /// How the input is normalised
        normalization: Normalization,
        /// Test-time augmentation
        tta: Tta,
        /// How patch predictions are blended
        blend: Blend,
        /// How the blended output is finalised
        finalize: Finalize,
        /// Patch overlap as a fraction of the patch
        overlap: f64,
    },
    /// hercUNet's `jacobi_refine` (HercUNet v0)
    Hercunet {
        /// The precision the network runs in
        precision: Precision,
        /// How the input is normalised
        normalization: Normalization,
        /// Names of the pipeline's input channels
        inputs: Vec<String>,
        /// Test-time augmentation
        tta: Tta,
        /// How patch predictions are blended
        blend: Blend,
        /// How the blended output is finalised
        finalize: Finalize,
        /// Patch overlap as a fraction of the patch
        overlap: f64,
        /// Number of Jacobi refinement passes
        passes: usize,
    },
    /// villa's `lasagna/qualzz_inference.py` (AFV v2), its seg head only
    AfvV2 {
        /// The precision the network runs in
        precision: Precision,
        /// How the input is normalised
        normalization: Normalization,
        /// How tile predictions are blended
        blend: Blend,
        /// How the blended output is finalised
        finalize: Finalize,
    },
    /// The network only, for a pipeline with no engine (Socratic F0)
    ProgramOnly,
}

impl Recipe {
    /// The pipeline's name, as the sidecar spells it
    pub fn name(&self) -> &'static str {
        match self {
            Self::VesuviusPredict { .. } => "vesuvius-predict",
            Self::FiberTool { .. } => "fiber-tool",
            Self::Hercunet { .. } => "hercunet",
            Self::AfvV2 { .. } => "afv-v2",
            Self::ProgramOnly => "program-only",
        }
    }
}

/// The export's contract
#[derive(Clone, Debug, Deserialize)]
pub struct Contract {
    /// The sidecar's format string, [`FORMAT`]
    pub format: String,
    /// The exported model's name
    pub name: String,
    /// Patch edge lengths `[z, y, x]` in voxels
    pub patch_size: [usize; 3],
    /// Input channels of the network
    pub in_channels: usize,
    /// Classes the head outputs
    pub num_classes: usize,
    /// The pipeline the weights run in
    pub recipe: Recipe,
    /// The layer program
    pub program: Program,
    /// Every exported tensor and its shape
    pub keys: BTreeMap<String, Vec<usize>>,
    /// Alias names mapped to the exported tensor each stands for
    pub aliases: BTreeMap<String, String>,
    /// Checkpoint tensors the inference graph never reads, with the reason
    pub dropped: BTreeMap<String, String>,
    /// The exporter's count, `2 x` [`Program::macs`]
    pub flops_per_patch: u64,
}

/// The `afv-v2` recipe [`afv2`] runs, checked
#[derive(Clone, Debug)]
pub struct AfvV2Recipe {
    /// `normalize_ct`'s floor on the tile's std
    pub std_floor: f32,
    /// The tile border the ramp zeroes
    pub border: usize,
    /// One axis of the tile weight: 0 on the border, 1 on the core
    pub ramp: Vec<f32>,
}

impl Contract {
    /// Parses and checks the sidecar: the format, the program's geometry and channels for this patch and
    /// input, the head's classes, the tensors the program reads against the exported keys (each exactly
    /// once, nothing else but the recipe's tensors), the recipe's own tables, and the FLOP count
    pub fn from_sidecar(sidecar: &serde_json::Value) -> Result<Self> {
        let c: Contract =
            serde_json::from_value(sidecar.clone()).map_err(|e| Unsupported(format!("not an {FORMAT} export: {e}")))?;
        ensure_supported!(c.format == FORMAT, "sidecar format {:?}, expected {FORMAT}", c.format);
        c.program.levels(c.patch_size, c.in_channels)?;
        ensure_supported!(
            c.program.head.cout == c.num_classes && c.num_classes >= 2,
            "head of {} channels for {} classes",
            c.program.head.cout,
            c.num_classes
        );
        let macs = c.program.macs(c.patch_size);
        ensure_supported!(
            2 * macs == c.flops_per_patch,
            "the exporter counts {} FLOPs a patch, the program {}",
            c.flops_per_patch,
            2 * macs
        );
        c.check_keys()?;
        match &c.recipe {
            Recipe::VesuviusPredict { normalization: Normalization::Ct(ct), .. } => ct.check()?,
            Recipe::Hercunet { normalization: Normalization::Ct(ct), .. } => ct.check()?,
            _ => {}
        }
        Ok(c)
    }

    /// The tensor names the program reads, in program order, each once
    pub fn program_tensors(&self) -> Vec<&str> {
        fn conv<'a>(out: &mut Vec<&'a str>, c: &'a ConvSpec) {
            out.push(c.w.as_str());
            out.extend(c.b.as_deref());
            if let Some(n) = &c.norm {
                out.extend([n.w.as_str(), n.b.as_str()]);
            }
        }
        let p = &self.program;
        let mut out = Vec::new();
        for s in &p.stem {
            conv(&mut out, s);
        }
        for b in p.stages.iter().flatten() {
            conv(&mut out, &b.conv1);
            conv(&mut out, &b.conv2);
            if let Some(pj) = &b.skip.proj {
                conv(&mut out, pj);
            }
        }
        for d in &p.decoder {
            conv(&mut out, &d.transp);
            for c in &d.convs {
                conv(&mut out, c);
            }
        }
        conv(&mut out, &p.head);
        out
    }

    /// The recipe's own tensors
    fn recipe_tensors(&self) -> Vec<&str> {
        match &self.recipe {
            Recipe::VesuviusPredict { blend: Blend::GaussianMap { tensor, .. }, .. }
            | Recipe::Hercunet { blend: Blend::GaussianMap { tensor, .. }, .. } => vec![tensor.as_str()],
            _ => Vec::new(),
        }
    }

    /// Fails unless the program reads every exported tensor but the recipe's exactly once, and reads nothing
    /// else; aliases must point at exported tensors
    fn check_keys(&self) -> Result<()> {
        let mut seen = BTreeSet::new();
        for t in self.program_tensors() {
            ensure_supported!(seen.insert(t), "the program reads {t} twice");
            ensure_supported!(self.keys.contains_key(t), "the program reads {t}, which the export does not hold");
        }
        for t in self.recipe_tensors() {
            ensure_supported!(seen.insert(t), "recipe tensor {t} is also a network tensor");
            ensure_supported!(self.keys.contains_key(t), "recipe tensor {t} missing from the export");
        }
        let unread: Vec<&String> = self.keys.keys().filter(|k| !seen.contains(k.as_str())).collect();
        ensure_supported!(unread.is_empty(), "exported tensors nothing reads: {unread:?}");
        for (alias, target) in &self.aliases {
            ensure_supported!(self.keys.contains_key(target), "alias {alias} of {target}, which is not exported");
        }
        Ok(())
    }
}

impl Contract {
    /// The `afv-v2` recipe with what [`afv2`] implements: float32 with TF32 math, the per-tile z-score, a flat ramp
    /// (no linear edges, `overlap == 2 * border`) on a cubic tile, the `semantic` and `tracing` uint8 outputs, one
    /// input channel and four classes (background, vertical, horizontal, intersection), a program of cubic kernels
    /// with strides of 1 or 2 per axis; anything else is [`Unsupported`]
    pub fn afv_v2(&self) -> Result<AfvV2Recipe> {
        let Recipe::AfvV2 { precision, normalization, blend, finalize } = &self.recipe else {
            return Err(Unsupported(format!("recipe {}, the AFV v2 engine runs afv-v2", self.recipe.name())).into());
        };
        let Normalization::TileZscore { std_floor } = normalization else {
            return Err(Unsupported(format!("normalization {normalization:?}, only tile_zscore")).into());
        };
        let Blend::Ramp { border, overlap } = blend else {
            return Err(Unsupported(format!("blend {blend:?}, only ramp")).into());
        };
        ensure_supported!(*precision == Precision::Tf32, "precision {precision:?}, the AFV v2 engine runs tf32");
        ensure_supported!(finalize.mode == "semantic_tracing_u8", "finalize {}", finalize.mode);
        ensure_supported!(*std_floor > 0.0 && std_floor.is_finite(), "std floor {std_floor}");
        let p = self.patch_size[0];
        ensure_supported!(self.patch_size.iter().all(|&v| v == p), "patch {:?}, only cubic", self.patch_size);
        // lasagna's linear edges (np.linspace) are not implemented; with overlap 2 * border there are none
        ensure_supported!(
            *overlap == 2 * border && 2 * border < p,
            "ramp border {border} overlap {overlap} on a {p} tile; only a flat ramp (overlap = 2 x border)"
        );
        ensure_supported!(self.in_channels == 1, "{} input channels, the AFV v2 engine takes 1", self.in_channels);
        ensure_supported!(
            self.num_classes == 4,
            "{} classes; AFV v2's seg head has background, vertical, horizontal and intersection",
            self.num_classes
        );
        self.program.check_axes()?;
        let ramp = (0..p).map(|i| if i >= *border && i < p - border { 1.0 } else { 0.0 }).collect();
        Ok(AfvV2Recipe { std_floor: *std_floor as f32, border: *border, ramp })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> serde_json::Value {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/nnunet").join(format!("{name}.json"));
        serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
            .expect("fixture JSON")
    }

    /// AFV v2's export runs as `afv-v2`: AFV v1's backbone at its FLOPs, float32 with TF32 math, the tile z-score
    /// floored at 10, a flat ramp (border 16, 96 core voxels of weight 1), the three auxiliary heads dropped with a
    /// reason; AFV v1's and m7's exports parse and refuse `afv-v2`
    #[test]
    fn afv_v2_export_checks() {
        let v2 = Contract::from_sidecar(&fixture("afv2")).expect("afv2");
        let v1 = Contract::from_sidecar(&fixture("afv")).expect("afv");
        assert_eq!(v2.recipe.name(), "afv-v2");
        assert_eq!((v2.num_classes, v2.flops_per_patch), (4, 1_436_857_401_344));
        assert_eq!(v2.flops_per_patch, v1.flops_per_patch);
        let levels = v2.program.levels(v2.patch_size, 1).expect("levels");
        assert_eq!(levels, v1.program.levels(v1.patch_size, 1).expect("levels"));
        // AFV's last stage keeps x
        assert_eq!(levels.last().map(|l| l.0), Some([2, 2, 4]));
        let r = v2.afv_v2().expect("afv2 runs as afv-v2");
        assert_eq!((r.std_floor, r.border, r.ramp.len()), (10.0, 16, 128));
        assert_eq!(r.ramp.iter().sum::<f32>(), 96.0);
        assert_eq!((r.ramp[15], r.ramp[16], r.ramp[111], r.ramp[112]), (0.0, 1.0, 1.0, 0.0));
        for head in ["centre", "affinity", "direction"] {
            assert!(v2.dropped.keys().any(|k| k.starts_with(&format!("{head}."))), "{head} not dropped");
        }
        for name in ["afv", "m7"] {
            let c = Contract::from_sidecar(&fixture(name)).expect(name);
            assert!(crate::error::is_unsupported(&c.afv_v2().unwrap_err()), "{name}");
        }

        // a ramp with linear edges, another precision, no std floor or another output is refused
        for (path, value) in [
            ("/recipe/blend/overlap", serde_json::json!(48)),
            ("/recipe/precision", serde_json::json!("bf16")),
            ("/recipe/normalization/std_floor", serde_json::json!(0.0)),
            ("/recipe/finalize/mode", serde_json::json!("fiber_probs_u8")),
        ] {
            let mut bad = fixture("afv2");
            *bad.pointer_mut(path).expect(path) = value;
            let c = Contract::from_sidecar(&bad).expect("parses");
            assert!(crate::error::is_unsupported(&c.afv_v2().unwrap_err()), "{path}");
        }
    }

    /// A sidecar whose program reads a tensor the export lacks, or leaves an exported tensor unread, or whose
    /// CT table disagrees with its parameters, is refused before any GPU work
    #[test]
    fn tensor_and_table_mismatches_refused() {
        let base = fixture("m7");
        let mut extra = base.clone();
        extra["keys"]["stray.weight"] = serde_json::json!([3]);
        let err = Contract::from_sidecar(&extra).unwrap_err();
        assert!(err.to_string().contains("stray.weight"), "{err}");

        let mut missing = base.clone();
        let head = missing["program"]["head"]["w"].as_str().expect("head").to_string();
        missing["keys"].as_object_mut().expect("keys").remove(&head);
        let err = Contract::from_sidecar(&missing).unwrap_err();
        assert!(err.to_string().contains(&head), "{err}");

        let mut table = base;
        table["recipe"]["normalization"]["table_f32_bits"][100] = serde_json::json!(0);
        let err = Contract::from_sidecar(&table).unwrap_err();
        assert!(err.to_string().contains("entry 100"), "{err}");
    }

    /// AFV v2's export loads as [`afv2::AfvV2`] loads it: every tensor taken exactly once by the float32 host net,
    /// the auxiliary heads absent. Reads the `afv-v2` export `VOLKIT_AFV2_EXPORT` names, and skips without one
    #[test]
    fn afv2_export_loads_every_tensor_once() {
        let path = std::path::PathBuf::from(std::env::var_os("VOLKIT_AFV2_EXPORT").unwrap_or_default());
        if !path.is_file() {
            eprintln!("skipped: no afv-v2 export at {}", path.display());
            return;
        }
        let (ck, sidecar) = crate::weights::Checkpoint::read(&path).expect("afv2 export");
        let contract = Contract::from_sidecar(&sidecar).expect("afv2 sidecar");
        contract.afv_v2().expect("afv-v2");
        assert!(contract.recipe_tensors().is_empty());
        crate::unet3d::model::HostNet::take(&ck, &contract.program, Precision::Tf32).expect("host net");
        let (tensors, _, params) = ck.finish().expect("every tensor read");
        assert_eq!(tensors, contract.keys.len());
        assert!(!contract.keys.keys().any(|k| ["centre.", "affinity.", "direction."].iter().any(|h| k.starts_with(h))));
        assert!(params > 1_000_000, "{params} parameters");
    }
}
