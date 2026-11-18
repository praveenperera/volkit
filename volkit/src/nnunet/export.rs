//! An nnU-Net ResEnc-L model folder mapped to the `inkrs-nnunet-v1` export (safetensors plus a JSON sidecar)
//! without torch, as the Python reference exporter maps it with its `afv-v2` recipe
//!
//! The Python exporter builds `dynamic_network_architectures`' `ResidualEncoderUNet` from the plans and walks the
//! module tree. Here the same layer program and state-dict names are derived from the plans' `arch_kwargs`
//! directly, following that package's construction (ResidualEncoder, StackedResidualBlocks of BasicBlockD,
//! StackedConvBlocks of ConvDropoutNormReLU, UNetDecoder). The checkpoint's tensors must match the derived state
//! dict name for name, shape for shape and in order, which is what `load_state_dict(strict=True)` checks, and
//! the tensors the checkpoint stores once must be the ones the network shares (`all_modules.*`, and the
//! decoder's reference to the encoder). Anything outside the op set and recipe AFV v2 needs is refused

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::pth::{Pth, TensorRef, Value as PyValue};
use crate::weights::Tensors;

/// The sidecar and safetensors `reader` format
const FORMAT: &str = "inkrs-nnunet-v1";
/// The network class the plans must name
const ARCH: &str = "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet";
/// The export's name
const NAME: &str = "afv2";
const AUX_REASON: &str = "auxiliary head (centre, affinity, direction), unused by the seg-only pipeline";
const DEEP_SUPERVISION_REASON: &str = "deep-supervision segmentation layer, unused at inference";

/// The op names a ResEnc-L plans file may give, by the dotted paths nnU-Net writes; any other is refused, and
/// none is ever imported
const CONV_OPS: [&str; 2] = ["torch.nn.modules.conv.Conv3d", "torch.nn.Conv3d"];
const NORM_OPS: [&str; 2] = ["torch.nn.modules.instancenorm.InstanceNorm3d", "torch.nn.InstanceNorm3d"];
const NONLINS: [&str; 2] = ["torch.nn.LeakyReLU", "torch.nn.modules.activation.LeakyReLU"];

/// One export: the unique float32 tensors, the names that alias them and the JSON sidecar
pub struct Export {
    /// The JSON sidecar `export_nnunet.py` writes beside the `.safetensors`: the inference contract
    pub sidecar: Value,
    /// Unique tensors in state-dict order
    tensors: Vec<Tensor>,
    /// Alias -> target in state-dict order, the order Python's `json.dumps(aliases)` writes
    aliases: Vec<(String, String)>,
}

struct Tensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Export {
    /// Reads an nnU-Net model folder without torch and maps it as the Python exporter's `afv-v2` recipe does
    pub fn afv_v2(dir: &Path) -> Result<Self> {
        let root = model_root(dir);
        let checkpoint = find_checkpoint(dir)?;
        let plans = read_json(&root.join("plans.json"))?;
        let dataset = read_json(&root.join("dataset.json"))?;
        let pth = Pth::open(&checkpoint)?;
        let source = Source::read(pth.payload()).wrap_err_with(|| format!("reading {}", checkpoint.display()))?;

        let labels =
            dataset.get("labels").and_then(Value::as_object).ok_or_else(|| eyre!("dataset.json has no labels"))?;
        ensure!(!labels.values().any(Value::is_array), "unsupported: region-based labels");
        // nnU-Net's LabelManager: one output per label, minus the ignore label
        let num_classes = labels.len() - usize::from(labels.contains_key("ignore"));
        let declared_channels = match dataset.get("channel_names") {
            Some(Value::Object(names)) => names.len(),
            Some(Value::Array(names)) => names.len(),
            _ => bail!("dataset.json has no channel_names"),
        };
        let stem = source
            .state
            .iter()
            .find(|(name, _)| name == "encoder.stem.convs.0.conv.weight")
            .ok_or_else(|| eyre!("checkpoint has no encoder.stem.convs.0.conv.weight"))?;
        let in_channels = *stem.1.shape().get(1).ok_or_else(|| eyre!("stem weight of shape {:?}", stem.1.shape()))?;

        let config = configuration(&plans, &source.configuration)?;
        let recipe = afv_v2_recipe(&plans, config, &dataset)?;
        let patch = patch_size(config)?;
        let network = Network::from_plans(&plans, &source.configuration, in_channels, num_classes)?;
        let mapped = network.map(&source.state)?;

        let mut dropped: Vec<(String, &str)> = source.auxiliary.iter().map(|k| (k.clone(), AUX_REASON)).collect();
        dropped.extend(mapped.dropped.iter().map(|k| (k.clone(), DEEP_SUPERVISION_REASON)));
        let tensors = mapped
            .unique
            .iter()
            .map(|(name, t)| Ok(Tensor { name: name.clone(), shape: t.shape().to_vec(), data: pth.tensor_f32(t)? }))
            .collect::<Result<Vec<_>>>()?;

        let keys: serde_json::Map<String, Value> = tensors.iter().map(|t| (t.name.clone(), json!(t.shape))).collect();
        let aliases: serde_json::Map<String, Value> =
            mapped.aliases.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
        let dropped: serde_json::Map<String, Value> = dropped.into_iter().map(|(k, v)| (k, json!(v))).collect();
        let sidecar = json!({
            "format": FORMAT,
            "name": NAME,
            "patch_size": patch,
            "in_channels": in_channels,
            "declared_channels": declared_channels,
            "num_classes": network.program.head.cout,
            "configuration": source.configuration,
            "trainer": source.trainer,
            "mirror_axes": source.mirror_axes,
            "recipe": recipe,
            "program": serde_json::to_value(&network.program)?,
            "checkpoint": checkpoint.to_str().ok_or_else(|| eyre!("checkpoint path is not UTF-8"))?,
            "checkpoint_sha256": sha256(&checkpoint)?,
            "keys": keys,
            "aliases": aliases,
            "dropped": dropped,
            "flops_per_patch": 2 * network.macs(patch)?,
        });
        Ok(Self { sidecar, tensors, aliases: mapped.aliases })
    }

    /// Writes `path` (.safetensors) and its .json sidecar (path.with_extension("json"))
    ///
    /// The safetensors header is written here rather than by the `safetensors` crate, which serialises its
    /// metadata map in random order; written this way the file matches the Python exporter's byte for byte. Both
    /// files go to a temporary name first, so a failed write leaves no partial export behind
    pub fn write(&self, path: &Path) -> Result<()> {
        let mut sorted: Vec<&Tensor> = self.tensors.iter().collect();
        // safetensors orders tensors by descending dtype alignment, then by name; all are float32 here
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        let mut header = format!(
            "{{\"__metadata__\":{{\"reader\":{},\"aliases\":{}}}",
            serde_json::to_string(FORMAT)?,
            serde_json::to_string(&py_json_dumps(&self.aliases))?
        );
        let mut offset = 0usize;
        for t in &sorted {
            let end = offset + 4 * t.data.len();
            let name = serde_json::to_string(&t.name)?;
            let shape = serde_json::to_string(&t.shape)?;
            write!(header, ",{name}:{{\"dtype\":\"F32\",\"shape\":{shape},\"data_offsets\":[{offset},{end}]}}")?;
            offset = end;
        }

        header.push('}');
        // the data starts 8-byte aligned, the header padded with spaces as safetensors pads it
        while header.len() % 8 != 0 {
            header.push(' ');
        }

        write_atomic(path, |w| {
            w.write_all(&(header.len() as u64).to_le_bytes())?;
            w.write_all(header.as_bytes())?;
            let mut buf = Vec::with_capacity(1 << 16);
            for t in &sorted {
                for chunk in t.data.chunks(1 << 14) {
                    buf.clear();
                    buf.extend(chunk.iter().flat_map(|v| v.to_le_bytes()));
                    w.write_all(&buf)?;
                }
            }

            Ok(())
        })?;

        let mut json = Vec::new();
        let formatter = serde_json::ser::PrettyFormatter::with_indent(b" ");
        self.sidecar.serialize(&mut serde_json::Serializer::with_formatter(&mut json, formatter))?;
        write_atomic(&path.with_extension("json"), |w| Ok(w.write_all(&json)?))
    }

    /// The unique tensors (name -> (shape, float32 data)) and aliases (alias -> target), for an in-memory load
    pub fn into_parts(self) -> (Tensors, BTreeMap<String, String>, Value) {
        let tensors = self.tensors.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
        (tensors, self.aliases.into_iter().collect(), self.sidecar)
    }
}

/// The directory holding plans.json and dataset.json: a `fold_*` directory's parent
fn model_root(dir: &Path) -> &Path {
    match (dir.file_name(), dir.parent()) {
        (Some(name), Some(parent)) if name.to_string_lossy().starts_with("fold_") => parent,
        _ => dir,
    }
}

/// The checkpoint villa's `load_model` picks: `fold_0/checkpoint_final.pth`, then `fold_0/checkpoint_best.pth`,
/// then the same names beside plans.json; a `fold_*` directory is searched itself
fn find_checkpoint(dir: &Path) -> Result<PathBuf> {
    let is_fold = dir.file_name().is_some_and(|name| name.to_string_lossy().starts_with("fold_"));
    let fold = if is_fold { dir.to_path_buf() } else { dir.join("fold_0") };
    let root = model_root(dir);
    let candidates = [
        fold.join("checkpoint_final.pth"),
        fold.join("checkpoint_best.pth"),
        root.join("checkpoint_final.pth"),
        root.join("checkpoint_best.pth"),
    ];
    let found = candidates.into_iter().find(|path| path.exists());
    found.ok_or_else(|| {
        eyre!("no checkpoint_final.pth or checkpoint_best.pth in {} or {}", fold.display(), root.display())
    })
}

fn read_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).wrap_err_with(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).wrap_err_with(|| format!("parsing {}", path.display()))
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).wrap_err_with(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = file.read(&mut buf).wrap_err_with(|| format!("reading {}", path.display()))?;
        if n == 0 {
            break;
        }

        hasher.update(&buf[..n]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

fn write_atomic(path: &Path, write: impl FnOnce(&mut BufWriter<File>) -> Result<()>) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".partial");
    let tmp = PathBuf::from(tmp);
    let file = File::create(&tmp).wrap_err_with(|| format!("creating {}", tmp.display()))?;
    let mut w = BufWriter::new(file);
    write(&mut w).and_then(|()| Ok(w.flush()?)).wrap_err_with(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, path).wrap_err_with(|| format!("renaming {} to {}", tmp.display(), path.display()))
}

/// Python's `json.dumps` of a str -> str dict with its defaults: `", "` and `": "` separators, ASCII only
fn py_json_dumps(pairs: &[(String, String)]) -> String {
    let mut out = String::from("{");
    for (i, (k, v)) in pairs.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }

        py_json_str(k, &mut out);
        out.push_str(": ");
        py_json_str(v, &mut out);
    }

    out.push('}');
    out
}

fn py_json_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                // ensure_ascii writes everything else as lowercase \uXXXX, astral characters as surrogate pairs
                let mut units = [0; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }

    out.push('"');
}

/// What the export reads from the pickled payload
struct Source {
    /// `network_weights` in pickled order
    state: Vec<(String, TensorRef)>,
    /// `auxiliary_weights` keys (AFV v2's centre, affinity and direction heads)
    auxiliary: Vec<String>,
    configuration: String,
    trainer: Option<String>,
    mirror_axes: Vec<i64>,
}

impl Source {
    fn read(payload: &PyValue) -> Result<Self> {
        ensure!(payload.as_dict().is_some(), "the checkpoint payload is not a dict");
        let weights = payload.get("network_weights").ok_or_else(|| eyre!("no network_weights"))?;
        let weights = weights.as_dict().ok_or_else(|| eyre!("network_weights is not a dict"))?;
        let state = weights
            .iter()
            .map(|(k, v)| match (k, v) {
                (PyValue::Str(name), PyValue::Tensor(t)) => Ok((name.clone(), t.clone())),
                (PyValue::Str(name), other) => bail!("network_weights.{name} is not a tensor: {other:?}"),
                (key, _) => bail!("network_weights key {key:?} is not a string"),
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some((name, _)) = state.iter().find(|(name, _)| name.starts_with("base.")) {
            bail!("unsupported: {name}: a `base.`-wrapped network (HercUNet) is not an AFV v2 checkpoint");
        }

        let auxiliary = match payload.get("auxiliary_weights") {
            None | Some(PyValue::None) => Vec::new(),
            Some(PyValue::Dict(entries)) => entries
                .iter()
                .map(|(k, _)| k.as_str().map(str::to_string).ok_or_else(|| eyre!("auxiliary key {k:?}")))
                .collect::<Result<_>>()?,
            Some(other) => bail!("auxiliary_weights is {other:?}"),
        };
        // nnU-Net records the configuration in the trainer's init arguments; older checkpoints lack it
        let configuration = match payload.get("init_args") {
            None => None,
            Some(init) => {
                ensure!(init.as_dict().is_some(), "init_args is not a dict");
                init.get("configuration")
            }
        };
        let configuration = match configuration {
            None => "3d_fullres".to_string(),
            Some(PyValue::Str(s)) => s.clone(),
            Some(other) => bail!("init_args.configuration is {other:?}"),
        };
        let trainer = match payload.get("trainer_name") {
            None | Some(PyValue::None) => None,
            Some(PyValue::Str(s)) => Some(s.clone()),
            Some(other) => bail!("trainer_name is {other:?}"),
        };
        let mirror_axes = match payload.get("inference_allowed_mirroring_axes") {
            None | Some(PyValue::None) => Vec::new(),
            Some(PyValue::List(items) | PyValue::Tuple(items)) => {
                items.iter().map(|v| v.as_int().ok_or_else(|| eyre!("mirror axis {v:?}"))).collect::<Result<_>>()?
            }
            Some(other) => bail!("inference_allowed_mirroring_axes is {other:?}"),
        };
        Ok(Self { state, auxiliary, configuration, trainer, mirror_axes })
    }
}

fn configuration<'a>(plans: &'a Value, name: &str) -> Result<&'a Value> {
    plans.get("configurations").and_then(|c| c.get(name)).ok_or_else(|| eyre!("plans have no configuration {name}"))
}

fn patch_size(config: &Value) -> Result<[usize; 3]> {
    let patch = config.get("patch_size").cloned().ok_or_else(|| eyre!("configuration has no patch_size"))?;
    serde_json::from_value(patch).wrap_err("patch_size is not three sizes")
}

/// The afv-v2 recipe and its preconditions: villa lasagna/qualzz_inference.py with AFV v2's model.py, float32
/// under PyTorch's defaults (TF32 convs), per-tile z-score, lasagna's ramp blend, seg head only
fn afv_v2_recipe(plans: &Value, config: &Value, dataset: &Value) -> Result<Value> {
    let want = json!({"background": 0, "vt-fiber": 1, "hz-fiber": 2, "intersection": 3});
    let labels = dataset.get("labels");
    ensure!(labels == Some(&want), "unsupported: afv-v2 recipe needs the classes {want}, the dataset gives {labels:?}");
    let schemes = config.get("normalization_schemes").cloned().unwrap_or(json!([]));
    ensure!(
        schemes == json!(["ZScoreNormalization"]),
        "unsupported: afv-v2 needs ZScoreNormalization, plans give {schemes}"
    );
    let transpose = plans.get("transpose_forward");
    ensure!(
        transpose == Some(&json!([0, 1, 2])),
        "unsupported: afv-v2 needs no axis transposition, plans give {transpose:?}"
    );
    Ok(json!({
        "pipeline": "afv-v2",
        "precision": "tf32",
        "normalization": {"mode": "tile_zscore", "std_floor": 10.0},
        "blend": {"mode": "ramp", "border": 16, "overlap": 32},
        "finalize": {"mode": "semantic_tracing_u8"},
    }))
}

/// `arch_kwargs` of a ResEnc plan; any key beyond these (a bottleneck block, a custom stem) is refused
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchKwargs {
    n_stages: usize,
    features_per_stage: Vec<usize>,
    conv_op: String,
    kernel_sizes: Vec<[usize; 3]>,
    strides: Vec<[usize; 3]>,
    n_blocks_per_stage: Vec<usize>,
    n_conv_per_stage_decoder: Vec<usize>,
    conv_bias: bool,
    norm_op: String,
    norm_op_kwargs: Option<NormKwargs>,
    dropout_op: Option<String>,
    /// Unused without a dropout op, as in the network
    #[serde(rename = "dropout_op_kwargs")]
    _dropout_op_kwargs: Option<Value>,
    nonlin: String,
    nonlin_kwargs: Option<NonlinKwargs>,
}

/// InstanceNorm3d's arguments, with torch's defaults
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NormKwargs {
    #[serde(default = "default_eps")]
    eps: f64,
    #[serde(default)]
    affine: bool,
}

impl Default for NormKwargs {
    fn default() -> Self {
        Self { eps: default_eps(), affine: false }
    }
}

fn default_eps() -> f64 {
    1e-5
}

/// LeakyReLU's arguments, with torch's defaults
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NonlinKwargs {
    #[serde(default = "default_slope")]
    negative_slope: f64,
    /// Changes no result
    #[serde(default, rename = "inplace")]
    _inplace: bool,
}

fn default_slope() -> f64 {
    0.01
}

/// One convolution of the program; a transposed conv has the same fields
#[derive(Debug, Clone, Serialize)]
struct Conv {
    w: String,
    b: Option<String>,
    cin: usize,
    cout: usize,
    kernel: [usize; 3],
    stride: [usize; 3],
}

#[derive(Debug, Serialize)]
struct Norm {
    w: String,
    b: String,
    eps: f64,
}

/// One ConvDropoutNormReLU: conv, affine instance norm, optional LeakyReLU
#[derive(Debug, Serialize)]
struct Unit {
    #[serde(flatten)]
    conv: Conv,
    norm: Norm,
    leaky: Option<f64>,
}

#[derive(Debug, Serialize)]
struct Skip {
    pool: [usize; 3],
    proj: Option<Unit>,
}

/// One BasicBlockD
#[derive(Debug, Serialize)]
struct Block {
    conv1: Unit,
    conv2: Unit,
    skip: Skip,
    leaky: f64,
}

#[derive(Debug, Serialize)]
struct DecoderStage {
    transp: Conv,
    convs: Vec<Unit>,
}

/// The layer program, serialised as the sidecar's `program`
#[derive(Debug, Serialize)]
struct Program {
    stem: Vec<Unit>,
    stages: Vec<Vec<Block>>,
    decoder: Vec<DecoderStage>,
    head: Conv,
}

/// One state-dict entry of the network
#[derive(Debug, Clone)]
struct Param {
    name: String,
    shape: Vec<usize>,
    /// The first name of the parameter this one shares, for `all_modules.*` and `decoder.encoder.*`
    alias_of: Option<String>,
}

/// The network the plans describe: its program and its state dict in `state_dict()` order
struct Network {
    program: Program,
    params: Vec<Param>,
    /// Deep-supervision segmentation layers, built but never run
    seg_layers: usize,
}

/// The checkpoint mapped onto the network
struct Mapped<'a> {
    unique: Vec<(String, &'a TensorRef)>,
    aliases: Vec<(String, String)>,
    dropped: Vec<String>,
}

impl Network {
    fn from_plans(plans: &Value, configuration_name: &str, in_channels: usize, num_classes: usize) -> Result<Self> {
        let arch =
            configuration(plans, configuration_name)?.get("architecture").ok_or_else(|| eyre!("no architecture"))?;
        let class = arch.get("network_class_name").and_then(Value::as_str);
        ensure!(class == Some(ARCH), "unsupported: network {class:?}, only {ARCH}");
        let kwargs = arch.get("arch_kwargs").cloned().ok_or_else(|| eyre!("no arch_kwargs"))?;
        let kw: ArchKwargs = serde_json::from_value(kwargs).wrap_err("unsupported arch_kwargs")?;

        ensure!(CONV_OPS.contains(&kw.conv_op.as_str()), "unsupported: conv_op {}", kw.conv_op);
        ensure!(NORM_OPS.contains(&kw.norm_op.as_str()), "unsupported: norm_op {}", kw.norm_op);
        ensure!(NONLINS.contains(&kw.nonlin.as_str()), "unsupported: nonlin {}", kw.nonlin);
        ensure!(kw.dropout_op.is_none(), "unsupported: dropout {:?}", kw.dropout_op);
        let norm = kw.norm_op_kwargs.unwrap_or_default();
        ensure!(norm.affine, "unsupported: InstanceNorm3d without affine");
        let slope = kw.nonlin_kwargs.map_or(default_slope(), |k| k.negative_slope);

        let n = kw.n_stages;
        ensure!(n >= 2, "unsupported: {n} stages");
        let per_stage =
            [kw.features_per_stage.len(), kw.kernel_sizes.len(), kw.strides.len(), kw.n_blocks_per_stage.len()];
        ensure!(per_stage.iter().all(|&len| len == n), "per-stage lists of lengths {per_stage:?} for {n} stages");
        ensure!(kw.n_conv_per_stage_decoder.len() == n - 1, "n_conv_per_stage_decoder needs {} entries", n - 1);
        let counts = kw.n_blocks_per_stage.iter().chain(&kw.n_conv_per_stage_decoder);
        ensure!(counts.copied().all(|c| c > 0), "a stage without blocks or convs");
        ensure!(in_channels > 0 && num_classes > 0, "{in_channels} input channels, {num_classes} classes");

        let mut b = Builder { params: Vec::new(), eps: norm.eps, slope, bias: kw.conv_bias };
        let (f, k, s) = (&kw.features_per_stage, &kw.kernel_sizes, &kw.strides);
        // ResidualEncoder: a one-conv stem at stage 0's width and kernel, stride 1, then the residual stages
        let stem = vec![b.unit("encoder.stem.convs.0", in_channels, f[0], k[0], [1; 3], UnitRole::Activated)?];
        let mut stages = Vec::with_capacity(n);
        let mut cin = f[0];
        for stage in 0..n {
            let mut blocks = Vec::with_capacity(kw.n_blocks_per_stage[stage]);
            for block in 0..kw.n_blocks_per_stage[stage] {
                let (block_in, stride) = if block == 0 { (cin, s[stage]) } else { (f[stage], [1; 3]) };
                let prefix = format!("encoder.stages.{stage}.blocks.{block}");
                blocks.push(b.block(&prefix, block_in, f[stage], k[stage], stride)?);
            }

            stages.push(blocks);
            cin = f[stage];
        }

        // UNetDecoder registers the encoder first, so its state dict repeats the encoder's under decoder.encoder
        let encoder: Vec<Param> = b.params.clone();
        for p in encoder {
            let target = p.alias_of.unwrap_or_else(|| p.name.clone());
            b.alias(format!("decoder.{}", p.name), &target, p.shape);
        }

        // then its stages, transposed convs and segmentation layers, each a ModuleList, from the bottleneck up
        let mut convs = Vec::with_capacity(n - 1);
        for d in 1..n {
            let skip = f[n - d - 1];
            let mut stage = Vec::with_capacity(kw.n_conv_per_stage_decoder[d - 1]);
            for c in 0..kw.n_conv_per_stage_decoder[d - 1] {
                let conv_in = if c == 0 { 2 * skip } else { skip };
                let prefix = format!("decoder.stages.{}.convs.{c}", d - 1);
                stage.push(b.unit(&prefix, conv_in, skip, k[n - d - 1], [1; 3], UnitRole::Activated)?);
            }

            convs.push(stage);
        }

        let mut transps = Vec::with_capacity(n - 1);
        for d in 1..n {
            transps.push(b.transposed(&format!("decoder.transpconvs.{}", d - 1), f[n - d], f[n - d - 1], s[n - d])?);
        }

        let mut seg = Vec::with_capacity(n - 1);
        for d in 1..n {
            seg.push(b.conv(
                &format!("decoder.seg_layers.{}", d - 1),
                f[n - d - 1],
                num_classes,
                [1; 3],
                [1; 3],
                true,
            )?);
        }

        let head = seg.pop().ok_or_else(|| eyre!("no segmentation layer"))?;
        let decoder = transps.into_iter().zip(convs).map(|(transp, convs)| DecoderStage { transp, convs }).collect();
        let program = Program { stem, stages, decoder, head };
        Ok(Self { program, params: b.params, seg_layers: n - 1 })
    }

    /// Matches the checkpoint's state dict to this network and keeps one tensor per storage, as the Python
    /// export keeps one per storage pointer: a tensor at the storage and offset of an earlier one of the same
    /// shape aliases that earlier one. The aliases this finds must be the ones the network shares
    fn map<'a>(&self, state: &'a [(String, TensorRef)]) -> Result<Mapped<'a>> {
        ensure!(
            state.len() == self.params.len(),
            "checkpoint holds {} network tensors, the plans' network {}",
            state.len(),
            self.params.len()
        );
        for ((name, t), p) in state.iter().zip(&self.params) {
            ensure!(name == &p.name, "checkpoint tensor {name} where the network has {}", p.name);
            ensure!(t.shape() == p.shape, "{name}: shape {:?}, the network's is {:?}", t.shape(), p.shape);
        }

        let mut by_ptr: HashMap<(&str, usize), usize> = HashMap::new();
        let mut unique = Vec::new();
        let mut aliases = Vec::new();
        for (i, ((name, t), p)) in state.iter().zip(&self.params).enumerate() {
            let ptr = (t.storage(), t.offset());
            let earlier = by_ptr.get(&ptr).map(|&j| &state[j]).filter(|(_, e)| e.shape() == t.shape());
            let target = earlier.map(|(n, _)| n.as_str());
            ensure!(
                target == p.alias_of.as_deref(),
                "{name}: the checkpoint shares it with {target:?}, the network with {:?}",
                p.alias_of
            );
            if let Some(target) = target {
                aliases.push((name.clone(), target.to_string()));
                continue;
            }

            by_ptr.insert(ptr, i);
            unique.push((name.clone(), t));
        }

        let used = self.program.tensor_names();
        let mut dropped = Vec::new();
        let deep_supervision: Vec<String> =
            (0..self.seg_layers - 1).map(|i| format!("decoder.seg_layers.{i}.")).collect();
        unique.retain(|(name, _)| {
            if used.contains(&name.as_str()) {
                return true;
            }

            dropped.push(name.clone());
            false
        });
        for name in &dropped {
            ensure!(
                deep_supervision.iter().any(|prefix| name.starts_with(prefix)),
                "tensor {name} is neither read by the program nor a known inference-unused tensor"
            );
        }

        let kept: Vec<&str> = unique.iter().map(|(name, _)| name.as_str()).collect();
        let missing: Vec<&&str> = used.iter().filter(|name| !kept.contains(*name)).collect();
        ensure!(missing.is_empty(), "the program reads tensors the network does not hold: {missing:?}");
        // an alias of a dropped tensor (the all_modules copy of a deep-supervision layer) goes with it
        aliases.retain(|(_, target)| kept.contains(&target.as_str()));
        Ok(Mapped { unique, aliases, dropped })
    }

    /// Multiply-accumulates of one forward of a `patch` (`program_macs`): every conv (output voxels x Cout x Cin
    /// x taps), every transposed conv (input voxels x Cin x Cout x taps) and the head
    fn macs(&self, patch: [usize; 3]) -> Result<u64> {
        let mut macs: u128 = 0;
        let mut shape = patch;
        for unit in &self.program.stem {
            shape = conv_macs(&unit.conv, shape, &mut macs);
        }

        let mut shapes = Vec::with_capacity(self.program.stages.len());
        for stage in &self.program.stages {
            for block in stage {
                if let Some(proj) = &block.skip.proj {
                    conv_macs(&proj.conv, std::array::from_fn(|a| shape[a] / block.skip.pool[a]), &mut macs);
                }

                shape = conv_macs(&block.conv1.conv, shape, &mut macs);
                shape = conv_macs(&block.conv2.conv, shape, &mut macs);
            }

            shapes.push(shape);
        }

        for (d, stage) in self.program.decoder.iter().enumerate() {
            let t = &stage.transp;
            macs += volume(shape) * (t.cin * t.cout) as u128 * volume(t.kernel);
            shape = shapes[shapes.len() - 2 - d];
            for unit in &stage.convs {
                shape = conv_macs(&unit.conv, shape, &mut macs);
            }
        }

        conv_macs(&self.program.head, shape, &mut macs);
        u64::try_from(macs).map_err(|_| eyre!("{macs} multiply-accumulates per patch"))
    }
}

/// Adds a conv's multiply-accumulates over an input of `shape` ("same" padding) and returns its output shape
fn conv_macs(spec: &Conv, shape: [usize; 3], macs: &mut u128) -> [usize; 3] {
    let out: [usize; 3] = std::array::from_fn(|a| shape[a].div_ceil(spec.stride[a]));
    *macs += volume(out) * (spec.cout * spec.cin) as u128 * volume(spec.kernel);
    out
}

fn volume(v: [usize; 3]) -> u128 {
    v.iter().map(|&x| x as u128).product()
}

impl Program {
    /// Every tensor the program reads
    fn tensor_names(&self) -> Vec<&str> {
        let mut out = Vec::new();
        collect_names(self, &mut out);
        out
    }
}

fn collect_names<'a>(program: &'a Program, out: &mut Vec<&'a str>) {
    fn conv<'a>(c: &'a Conv, out: &mut Vec<&'a str>) {
        out.push(&c.w);
        out.extend(c.b.as_deref());
    }

    fn unit<'a>(u: &'a Unit, out: &mut Vec<&'a str>) {
        conv(&u.conv, out);
        out.extend([u.norm.w.as_str(), u.norm.b.as_str()]);
    }

    program.stem.iter().for_each(|u| unit(u, out));
    for block in program.stages.iter().flatten() {
        unit(&block.conv1, out);
        unit(&block.conv2, out);
        if let Some(proj) = &block.skip.proj {
            unit(proj, out);
        }
    }

    for stage in &program.decoder {
        conv(&stage.transp, out);
        stage.convs.iter().for_each(|u| unit(u, out));
    }

    conv(&program.head, out);
}

/// How a ConvDropoutNormReLU is configured where the network builds one
#[derive(Clone, Copy)]
enum UnitRole {
    /// The plans' conv bias, then LeakyReLU: the stem, a block's conv1, the decoder's convs
    Activated,
    /// The plans' conv bias, no activation: a block's conv2
    Plain,
    /// BasicBlockD's skip projection: a bias-free 1x1 conv and norm
    Projection,
}

/// Builds the program and the state dict together, module by module in registration order
struct Builder {
    params: Vec<Param>,
    eps: f64,
    slope: f64,
    bias: bool,
}

impl Builder {
    fn param(&mut self, name: String, shape: Vec<usize>) {
        self.params.push(Param { name, shape, alias_of: None });
    }

    fn alias(&mut self, name: String, target: &str, shape: Vec<usize>) {
        self.params.push(Param { name, shape, alias_of: Some(target.to_string()) });
    }

    /// A Conv3d at `prefix` with "same" padding; only odd kernels at per-axis stride 1 or 2
    fn conv(
        &mut self,
        prefix: &str,
        cin: usize,
        cout: usize,
        kernel: [usize; 3],
        stride: [usize; 3],
        bias: bool,
    ) -> Result<Conv> {
        ensure!(kernel.iter().all(|k| k % 2 == 1), "unsupported: {prefix}: kernel {kernel:?}, odd kernels only");
        ensure!(stride.iter().all(|s| (1..=2).contains(s)), "unsupported: {prefix}: stride {stride:?}, 1 or 2 only");
        let w = format!("{prefix}.weight");
        self.param(w.clone(), [cout, cin].into_iter().chain(kernel).collect());
        let b = bias.then(|| format!("{prefix}.bias"));
        if let Some(b) = &b {
            self.param(b.clone(), vec![cout]);
        }

        Ok(Conv { w, b, cin, cout, kernel, stride })
    }

    /// A ConvTranspose3d with kernel = stride, the decoder's upsampling; its weight is (Cin, Cout, kernel)
    fn transposed(&mut self, prefix: &str, cin: usize, cout: usize, stride: [usize; 3]) -> Result<Conv> {
        ensure!(stride.iter().all(|s| (1..=2).contains(s)), "unsupported: {prefix}: stride {stride:?}, 1 or 2 only");
        ensure!(self.bias, "unsupported: {prefix}: transposed conv without bias");
        let w = format!("{prefix}.weight");
        let b = format!("{prefix}.bias");
        self.param(w.clone(), [cin, cout].into_iter().chain(stride).collect());
        self.param(b.clone(), vec![cout]);
        Ok(Conv { w, b: Some(b), cin, cout, kernel: stride, stride })
    }

    /// A ConvDropoutNormReLU: `conv` and `norm`, then `all_modules` (the same modules again in a Sequential)
    fn unit(
        &mut self,
        prefix: &str,
        cin: usize,
        cout: usize,
        kernel: [usize; 3],
        stride: [usize; 3],
        role: UnitRole,
    ) -> Result<Unit> {
        let (bias, activation) = match role {
            UnitRole::Activated => (self.bias, true),
            UnitRole::Plain => (self.bias, false),
            UnitRole::Projection => (false, false),
        };
        let conv = self.conv(&format!("{prefix}.conv"), cin, cout, kernel, stride, bias)?;
        let norm = Norm { w: format!("{prefix}.norm.weight"), b: format!("{prefix}.norm.bias"), eps: self.eps };
        self.param(norm.w.clone(), vec![cout]);
        self.param(norm.b.clone(), vec![cout]);
        let weight_shape = [cout, cin].into_iter().chain(kernel).collect();
        self.alias(format!("{prefix}.all_modules.0.weight"), &conv.w, weight_shape);
        if let Some(b) = &conv.b {
            self.alias(format!("{prefix}.all_modules.0.bias"), b, vec![cout]);
        }

        self.alias(format!("{prefix}.all_modules.1.weight"), &norm.w, vec![cout]);
        self.alias(format!("{prefix}.all_modules.1.bias"), &norm.b, vec![cout]);
        let leaky = activation.then_some(self.slope);
        Ok(Unit { conv, norm, leaky })
    }

    /// A BasicBlockD: conv1 (strided, LeakyReLU), conv2 (no activation), then the skip, which average-pools
    /// when strided (`skip.0`) and projects with a bias-free 1x1 conv + norm when the width changes
    fn block(
        &mut self,
        prefix: &str,
        cin: usize,
        cout: usize,
        kernel: [usize; 3],
        stride: [usize; 3],
    ) -> Result<Block> {
        let conv1 = self.unit(&format!("{prefix}.conv1"), cin, cout, kernel, stride, UnitRole::Activated)?;
        let conv2 = self.unit(&format!("{prefix}.conv2"), cout, cout, kernel, [1; 3], UnitRole::Plain)?;
        // the pool, when there is one, is skip.0 and holds no tensors, so the projection follows it
        let index = usize::from(stride != [1; 3]);
        let proj = if cin == cout {
            None
        } else {
            Some(self.unit(&format!("{prefix}.skip.{index}"), cin, cout, [1; 3], [1; 3], UnitRole::Projection)?)
        };
        Ok(Block { conv1, conv2, skip: Skip { pool: stride, proj }, leaky: self.slope })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashSet};

    use super::*;

    fn fixture(name: &str) -> Result<Value> {
        read_json(&Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/nnunet").join(name))
    }

    /// The plans alone give the Python export's program, FLOPs and full state dict
    #[test]
    fn program_from_plans_matches_fixture() -> Result<()> {
        let sidecar = fixture("afv2.json")?;
        let plans = fixture("afv2.plans.json")?;
        let network = Network::from_plans(&plans, "3d_fullres", 1, 4)?;
        assert_eq!(serde_json::to_value(&network.program)?, sidecar["program"]);
        assert_eq!(json!(2 * network.macs([128; 3])?), sidecar["flops_per_patch"]);

        // every name the network holds is exported, aliased or dropped as a deep-supervision layer
        let keys = sidecar["keys"].as_object().ok_or_else(|| eyre!("no keys"))?;
        let aliases = sidecar["aliases"].as_object().ok_or_else(|| eyre!("no aliases"))?;
        let dropped = sidecar["dropped"].as_object().ok_or_else(|| eyre!("no dropped"))?;
        let ours: BTreeSet<&str> = network.params.iter().map(|p| p.name.as_str()).collect();
        let theirs: BTreeSet<&str> = keys
            .keys()
            .chain(aliases.keys())
            .chain(dropped.keys().filter(|k| k.starts_with("decoder.")))
            .map(String::as_str)
            .collect();
        assert_eq!(ours, theirs);
        for p in &network.params {
            match (&p.alias_of, keys.get(&p.name)) {
                (None, Some(shape)) => assert_eq!(&json!(p.shape), shape, "{}", p.name),
                (Some(target), _) if aliases.contains_key(&p.name) => assert_eq!(aliases[&p.name], json!(target)),
                (alias_of, _) => assert!(dropped.contains_key(&p.name), "{} ({alias_of:?}) unaccounted", p.name),
            }
        }

        Ok(())
    }

    #[test]
    fn py_json_dumps_escapes_as_python() {
        // the escapes are assembled so the expected text holds them literally
        let u = concat!("\\", "u");
        let pairs = [("a\"\\\u{e9}".to_string(), "b\n\u{1f600}".to_string()), ("c".to_string(), "d".to_string())];
        let want = format!(r#"{{"a\"\\{u}00e9": "b\n{u}d83d{u}de00", "c": "d"}}"#);
        assert_eq!(py_json_dumps(&pairs), want);
    }

    /// The sidecar keys equal, `checkpoint` (the path as passed in) only when the model folder is the one the
    /// reference export was made from
    fn assert_sidecar_eq(ours: &Value, want: &Value, model: &Path) -> Result<()> {
        let (ours, want) = (ours.as_object(), want.as_object());
        let (ours, want) = (ours.ok_or_else(|| eyre!("sidecar"))?, want.ok_or_else(|| eyre!("reference sidecar"))?);
        assert_eq!(ours.keys().collect::<Vec<_>>(), want.keys().collect::<Vec<_>>());
        let same_folder = want["checkpoint"].as_str().map(Path::new).and_then(Path::parent) == Some(model);
        for (key, value) in want {
            if key == "checkpoint" && !same_folder {
                eprintln!("not comparing checkpoint: the reference export was made from another folder path");
                continue;
            }

            assert_eq!(&ours[key], value, "sidecar {key}");
        }

        Ok(())
    }

    /// The torch-free export equals the Python one: tensors, aliases, sidecar, and the file's bytes. Reads the
    /// model folder from VOLKIT_AFV2_MODEL and the reference `.safetensors` (its sidecar beside it) from
    /// VOLKIT_AFV2_EXPORT, writes to VOLKIT_TEST_OUT (default: the system temp directory)
    #[test]
    fn afv_v2_matches_python_export() -> Result<()> {
        let (Some(model), Some(reference)) =
            (std::env::var_os("VOLKIT_AFV2_MODEL"), std::env::var_os("VOLKIT_AFV2_EXPORT"))
        else {
            eprintln!("skipping: VOLKIT_AFV2_MODEL or VOLKIT_AFV2_EXPORT is unset");
            return Ok(());
        };
        let (model, python) = (PathBuf::from(model), PathBuf::from(reference));
        if !model.is_dir() || !python.is_file() {
            eprintln!("skipping: {} or {} is missing", model.display(), python.display());
            return Ok(());
        }

        let export = Export::afv_v2(&model)?;
        let want = read_json(&python.with_extension("json"))?;
        assert_sidecar_eq(&export.sidecar, &want, &model)?;

        let out = std::env::var_os("VOLKIT_TEST_OUT").map_or_else(std::env::temp_dir, PathBuf::from);
        fs::create_dir_all(&out)?;
        let path = out.join(format!("afv2-{}.safetensors", std::process::id()));
        export.write(&path)?;
        assert_sidecar_eq(&read_json(&path.with_extension("json"))?, &want, &model)?;

        let ours_file = File::open(&path)?;
        let theirs_file = File::open(&python)?;
        // SAFETY: test-only read-only maps of files nothing else writes during the test
        let (ours_map, theirs_map) = unsafe { (memmap2::Mmap::map(&ours_file)?, memmap2::Mmap::map(&theirs_file)?) };
        let ours_st = safetensors::SafeTensors::deserialize(&ours_map)?;
        let theirs_st = safetensors::SafeTensors::deserialize(&theirs_map)?;
        let names: HashSet<String> = theirs_st.names().into_iter().map(str::to_string).collect();
        assert_eq!(ours_st.names().into_iter().map(str::to_string).collect::<HashSet<_>>(), names);
        for name in &names {
            let (a, b) = (ours_st.tensor(name)?, theirs_st.tensor(name)?);
            assert_eq!((a.dtype(), a.shape()), (safetensors::Dtype::F32, b.shape()), "{name}");
            assert_eq!(b.dtype(), safetensors::Dtype::F32, "{name}");
            assert!(a.data() == b.data(), "{name}: data differs");
        }

        let (_, ours_meta) = safetensors::SafeTensors::read_metadata(&ours_map)?;
        let (_, theirs_meta) = safetensors::SafeTensors::read_metadata(&theirs_map)?;
        assert_eq!(ours_meta.metadata(), theirs_meta.metadata());
        let identical = ours_map[..] == theirs_map[..];
        eprintln!("safetensors byte-identical to the Python export: {identical}");
        assert!(identical, "the safetensors file differs from the Python export's bytes");

        let (tensors, aliases, _) = export.into_parts();
        assert_eq!(tensors.len(), want["keys"].as_object().map_or(0, |k| k.len()));
        assert_eq!(json!(aliases), want["aliases"]);
        fs::remove_file(&path)?;
        fs::remove_file(path.with_extension("json"))?;
        Ok(())
    }
}
