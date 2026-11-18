//! The 3D U-Net forward at batch 1, built from an export's layer program
//!
//! Each resolution level owns its activation buffers, sized for the widest tensor at that level:
//! two for block inputs and outputs (`x`, `y`), raw conv outputs (`r`, `r2`), the normalised conv1
//! output (`h`, which also holds the transposed conv's GEMM output in the decoder), the pooled skip
//! input (`pa`) and the skip projection (`rs`) where blocks need them, and the decoder's concat
//! buffer (`cat`, `[up, skip]`). The last block of each encoder stage writes its output twice, packed
//! for the next stage and into the second half of that level's `cat`, so the concat costs no copy

use std::collections::HashMap;
use std::rc::Rc;

use color_eyre::eyre::{Result, WrapErr, ensure};

use super::{ConvSpec, Precision, Program};
use crate::dnn::{Blas, ConvPlan, ConvShape, Dnn, Dtype, Pick};
use crate::gpu::{Arg, Buf, Gpu};
use crate::weights::Checkpoint;

/// The stem's input channels when cuDNN refuses a one-channel 3D conv
const STEM_PAD: usize = 8;
/// Upper bound on [`crate::unet3d`]'s InstanceNorm partial-sum chunks per tensor
const MAX_CHUNKS: usize = 4096;

/// What the forward needs from the caller
pub struct Ctx<'a> {
    /// The GPU and its stream
    pub gpu: &'a Gpu,
    /// The cuDNN handle
    pub dnn: &'a Dnn,
    /// The cuBLAS handle
    pub blas: &'a Blas,
    /// How convolution algorithms are chosen
    pub pick: Pick,
    /// Workspace limit per convolution, in bytes
    pub max_workspace: usize,
}

/// The glue kernels of one precision (volkit-kernels `unet3d`)
struct Kernels {
    partial: &'static str,
    apply: &'static str,
    apply_pool: &'static str,
    pool2: &'static str,
    up: &'static str,
    /// Pool and up-scatter with a factor of 1 or 2 per axis, for stages that are not 2 on every axis
    poolk: &'static str,
    upk: &'static str,
}

impl Kernels {
    fn of(precision: Precision) -> Self {
        match precision {
            Precision::Bf16 => Self {
                partial: "u3_partial",
                apply: "u3_apply",
                apply_pool: "u3_apply_pool",
                pool2: "u3_pool2",
                up: "u3_up",
                poolk: "u3_poolk",
                upk: "u3_upk",
            },
            Precision::Fp16 => Self {
                partial: "u3h_partial",
                apply: "u3h_apply",
                apply_pool: "u3h_apply_pool",
                pool2: "u3h_pool2",
                up: "u3h_up",
                poolk: "u3h_poolk",
                upk: "u3h_upk",
            },
            Precision::Tf32 => Self {
                partial: "u3f_partial",
                apply: "u3f_apply",
                apply_pool: "u3f_apply_pool",
                pool2: "u3f_pool2",
                up: "u3f_up",
                poolk: "u3f_poolk",
                upk: "u3f_upk",
            },
        }
    }
}

/// One conv's weights read from the checkpoint and laid out on the CPU: the weight channels-last
/// `[K, taps, C]` float32 (rounded at upload, after any channel padding), the bias rounded to the
/// precision, the norm's float32 affine
pub struct HostConv {
    /// Input channels
    pub cin: usize,
    /// Output channels
    pub cout: usize,
    w: Vec<f32>,
    bias: Option<Vec<f32>>,
    norm: Option<(Vec<f32>, Vec<f32>, f32)>,
}

impl HostConv {
    fn take(ck: &Checkpoint, spec: &ConvSpec, precision: Precision) -> Result<Self> {
        let (k, c) = (spec.cout, spec.cin);
        let taps: usize = spec.kernel.iter().product();
        let shape: Vec<usize> = [k, c].into_iter().chain(spec.kernel.iter().copied()).collect();
        let w = ck.take(&spec.w, &shape)?;
        let mut ktc = vec![0f32; k * taps * c];
        for ko in 0..k {
            for ci in 0..c {
                for t in 0..taps {
                    ktc[(ko * taps + t) * c + ci] = w[(ko * c + ci) * taps + t];
                }
            }
        }
        let bias = match &spec.b {
            Some(key) => Some(ck.take(key, &[k])?.iter().map(|&v| precision.round(v)).collect()),
            None => None,
        };
        let norm = match &spec.norm {
            Some(n) => Some((ck.take(&n.w, &[k])?, ck.take(&n.b, &[k])?, n.eps)),
            None => None,
        };
        Ok(Self { cin: c, cout: k, w: ktc, bias, norm })
    }

    /// The weight as `[K, taps, c_pad]` in the storage type, as bytes, the channels past `cin` zero
    fn stored(&self, precision: Precision, c_pad: usize) -> Vec<u8> {
        let taps = self.w.len() / (self.cout * self.cin);
        let mut out = vec![0f32; self.cout * taps * c_pad];
        for (row, src) in self.w.chunks_exact(self.cin).enumerate() {
            out[row * c_pad..row * c_pad + self.cin].copy_from_slice(src);
        }
        precision.encode(&out)
    }
}

/// One BasicBlockD's weights on the CPU
pub struct HostBlock {
    conv1: HostConv,
    conv2: HostConv,
    proj: Option<HostConv>,
}

/// One decoder stage's weights on the CPU: the transposed conv as a GEMM weight `[cin, taps * cout]`
/// (column `tap * cout + co`), its bias rounded, and the convs
pub struct HostDec {
    g: Vec<f32>,
    tb: Vec<f32>,
    tcin: usize,
    convs: Vec<HostConv>,
}

/// Every network tensor read from the checkpoint and laid out on the CPU, each taken once by name and
/// shape, so [`Checkpoint::finish`] after this proves the program reads the whole export
pub struct HostNet {
    precision: Precision,
    stem: Vec<HostConv>,
    blocks: Vec<HostBlock>,
    decs: Vec<HostDec>,
    /// The head's weight `[classes, cin]` and bias, rounded
    head_w: Vec<f32>,
    head_b: Vec<f32>,
    head_cin: usize,
}

impl HostNet {
    /// Takes the program's tensors from `ck`
    pub fn take(ck: &Checkpoint, prog: &Program, precision: Precision) -> Result<Self> {
        let stem = prog.stem.iter().map(|s| HostConv::take(ck, s, precision)).collect::<Result<_>>()?;
        let mut blocks = Vec::new();
        for b in prog.stages.iter().flatten() {
            blocks.push(HostBlock {
                conv1: HostConv::take(ck, &b.conv1, precision)?,
                conv2: HostConv::take(ck, &b.conv2, precision)?,
                proj: b.skip.proj.as_ref().map(|pj| HostConv::take(ck, pj, precision)).transpose()?,
            });
        }
        let mut decs = Vec::new();
        for ds in &prog.decoder {
            let t = &ds.transp;
            let (ci, co) = (t.cin, t.cout);
            let taps: usize = t.kernel.iter().product();
            let shape: Vec<usize> = [ci, co].into_iter().chain(t.kernel.iter().copied()).collect();
            let w = ck.take(&t.w, &shape)?;
            let mut g = vec![0f32; ci * taps * co];
            for i in 0..ci {
                for o in 0..co {
                    for tap in 0..taps {
                        g[i * taps * co + tap * co + o] = w[(i * co + o) * taps + tap];
                    }
                }
            }
            let key = t.b.as_ref().expect("checked by the contract");
            let tb = ck.take(key, &[co])?.iter().map(|&v| precision.round(v)).collect();
            let convs = ds.convs.iter().map(|c| HostConv::take(ck, c, precision)).collect::<Result<_>>()?;
            decs.push(HostDec { g, tb, tcin: ci, convs });
        }
        let h = &prog.head;
        let head_w = ck.take(&h.w, &[h.cout, h.cin, 1, 1, 1])?.iter().map(|&v| precision.round(v)).collect();
        let key = h.b.as_ref().expect("checked by the contract");
        let head_b = ck.take(key, &[h.cout])?.iter().map(|&v| precision.round(v)).collect();
        Ok(Self { precision, stem, blocks, decs, head_w, head_b, head_cin: h.cin })
    }
}

/// An InstanceNorm's float32 affine
struct Norm {
    w: Buf,
    b: Buf,
    eps: f32,
}

/// A planned conv: channels-last weight `[K, kd, kh, kw, C]` in the storage type, its bias rounded to it,
/// the norm and activation after it
struct Conv {
    plan: Rc<ConvPlan>,
    w: Buf,
    bias: Option<Buf>,
    norm: Option<Norm>,
    leaky: Option<f32>,
    cout: usize,
}

struct Planner<'a> {
    ctx: &'a Ctx<'a>,
    dtype: Dtype,
    precision: Precision,
    plans: HashMap<ConvShape, Rc<ConvPlan>>,
}

impl Planner<'_> {
    fn plan(&mut self, shape: ConvShape) -> Result<Rc<ConvPlan>> {
        if let Some(p) = self.plans.get(&shape) {
            return Ok(p.clone());
        }
        let c = self.ctx;
        let plan = Rc::new(ConvPlan::typed(c.gpu, c.dnn, shape.clone(), self.dtype, c.pick, c.max_workspace)?);
        self.plans.insert(shape, plan.clone());
        Ok(plan)
    }

    /// `spec` with weights `host` on an input of `dims` (z, y, x) with `c_pad >= spec.cin` channels
    fn conv(&mut self, host: &HostConv, spec: &ConvSpec, dims: [usize; 3], c_pad: usize) -> Result<Conv> {
        let gpu = self.ctx.gpu;
        let (k, ks) = (spec.cout, spec.kernel[0]);
        let stride: Vec<i32> = spec.stride.iter().map(|&s| s as i32).collect();
        let shape = ConvShape::same_axes(1, c_pad as i32, k as i32, &dims.map(|d| d as i32), ks as i32, &stride);
        let plan = self.plan(shape)?;
        let bias = host.bias.as_ref().map(|b| gpu.upload(b)).transpose()?;
        let norm = match &host.norm {
            Some((w, b, eps)) => Some(Norm { w: gpu.upload(w)?, b: gpu.upload(b)?, eps: *eps }),
            None => None,
        };
        Ok(Conv { plan, w: gpu.upload(&host.stored(self.precision, c_pad))?, bias, norm, leaky: spec.leaky, cout: k })
    }
}

/// One BasicBlockD
struct Block {
    conv1: Conv,
    conv2: Conv,
    /// The skip's average-pool factor per axis, when it pools
    pool: Option<[usize; 3]>,
    proj: Option<Conv>,
    leaky: f32,
    /// Level of the block's input and output
    lin: usize,
    lout: usize,
}

/// One decoder stage, writing level `level`
struct Dec {
    /// GEMM weight `[cin, taps * cout]` in the storage type, column `tap * cout + co`
    tw: Buf,
    /// The transposed conv's kernel (= stride) per axis
    factor: [usize; 3],
    tb: Buf,
    tcin: usize,
    convs: Vec<Conv>,
    level: usize,
}

/// One resolution level's buffers
struct Level {
    /// z, y, x
    dims: [usize; 3],
    x: Buf,
    y: Buf,
    r: Buf,
    r2: Buf,
    h: Buf,
    pa: Option<Buf>,
    rs: Option<Buf>,
    cat: Option<Buf>,
}

/// The head: weight `[classes, cin]` and bias, rounded to the precision, as float32
pub struct Head<'a> {
    /// The last decoder output `[p^3, cin]` in the storage type
    pub x: &'a Buf,
    /// Input channels
    pub cin: usize,
    /// Output classes
    pub classes: usize,
    /// The weight `[classes, cin]`
    pub w: &'a Buf,
    /// The bias `[classes]`
    pub b: &'a Buf,
}

/// The network on the GPU, at batch 1
pub struct Model {
    /// Patch edge length in voxels; the model runs `patch^3` voxels
    pub patch: usize,
    /// The precision the network runs in
    pub precision: Precision,
    /// Channel stride of the input buffer [`Model::input`] (the stem's channels, or 8 for a padded stem)
    pub in_c: usize,
    kn: Kernels,
    stem: Vec<Conv>,
    blocks: Vec<Block>,
    decs: Vec<Dec>,
    head_w: Buf,
    head_b: Buf,
    head_c: usize,
    head_classes: usize,
    levels: Vec<Level>,
    /// The normalised patch, `[p^3, in_c]` in the storage type
    pub input: Buf,
    partial: Buf,
    stats: [Buf; 3],
    ws: Buf,
    /// (level, which buffer: 'x' or 'y') of the last decoder output, the head's input. The last
    /// stage ends in `y` after an odd number of convs and in `x` after an even one, so the head
    /// must read the buffer the forward wrote rather than assume one
    out: (usize, char),
}

impl Model {
    /// Plans every conv of `prog` (cubic patch of edge `p`, strides of 1 or 2 per axis), uploads `host`'s
    /// weights and allocates the buffers
    pub fn new(ctx: &Ctx, host: &HostNet, prog: &Program, p: usize) -> Result<Self> {
        let gpu = ctx.gpu;
        super::check_device(gpu.inner.compute_capability(), gpu.inner.ptx_tier())?;
        prog.check_axes()?;
        let precision = host.precision;
        let dtype = match precision {
            Precision::Bf16 => Dtype::Bf16,
            Precision::Fp16 => Dtype::Half,
            // float32 tensors with TF32 math, as PyTorch's defaults run float32 convolutions
            Precision::Tf32 => Dtype::Float { tf32: true },
        };
        let mut planner = Planner { ctx, dtype, precision, plans: HashMap::new() };

        // stem: its input channels, or padded when cuDNN refuses one channel
        let mut stem = Vec::new();
        let mut in_c = prog.stem.first().map_or(1, |s| s.cin);
        for (i, (s, hs)) in prog.stem.iter().zip(&host.stem).enumerate() {
            let c_pad = if i == 0 && s.cin == 1 {
                match planner.plan(ConvShape::same(1, 1, s.cout as i32, &[p as i32; 3], s.kernel[0] as i32, 1)) {
                    Ok(_) => 1,
                    Err(e) => {
                        eprintln!("unet3d: stem conv on 1 channel: {e:#}; padding to {STEM_PAD}");
                        in_c = STEM_PAD;
                        STEM_PAD
                    }
                }
            } else {
                s.cin
            };
            stem.push(planner.conv(hs, s, [p; 3], c_pad)?);
        }

        // encoder
        let mut blocks = Vec::new();
        let (mut dims, mut level) = ([p; 3], 0usize);
        let mut edges = vec![dims];
        // widest tensor per level, and which levels need the skip buffers
        let mut cmax: Vec<usize> = vec![prog.stem.last().map_or(1, |s| s.cout)];
        let mut need_pa: Vec<usize> = vec![0];
        let mut need_rs = vec![false];
        for (b, hb) in prog.stages.iter().flatten().zip(&host.blocks) {
            let lin = level;
            let stride = [b.conv1.stride[0], b.conv1.stride[1], b.conv1.stride[2]];
            if stride != [1; 3] {
                ensure!(
                    (0..3).all(|a| dims[a].is_multiple_of(stride[a])),
                    "stride {stride:?} does not divide {dims:?}"
                );
                dims = std::array::from_fn(|a| dims[a] / stride[a]);
                level += 1;
                edges.push(dims);
                cmax.push(0);
                need_pa.push(0);
                need_rs.push(false);
            }
            let lout = level;
            let conv1 = planner.conv(&hb.conv1, &b.conv1, edges[lin], b.conv1.cin)?;
            let conv2 = planner.conv(&hb.conv2, &b.conv2, dims, b.conv2.cin)?;
            let proj = match (&b.skip.proj, &hb.proj) {
                (Some(pj), Some(hp)) => Some(planner.conv(hp, pj, dims, pj.cin)?),
                _ => None,
            };
            let pool = Some(b.skip.pool.axes()).filter(|&a| a != [1; 3]);
            if pool.is_some() {
                need_pa[lout] = need_pa[lout].max(b.conv1.cin);
            }
            need_rs[lout] |= proj.is_some();
            cmax[lin] = cmax[lin].max(b.conv1.cin);
            cmax[lout] = cmax[lout].max(b.conv2.cout);
            blocks.push(Block { conv1, conv2, pool, proj, leaky: b.leaky, lin, lout });
        }
        let n_levels = level + 1;
        ensure!(prog.decoder.len() + 1 == n_levels, "{} decoder stages for {n_levels} levels", prog.decoder.len());

        // decoder
        let mut decs = Vec::new();
        for (d, (ds, hd)) in prog.decoder.iter().zip(&host.decs).enumerate() {
            let lv = n_levels - 2 - d;
            let co = ds.transp.cout;
            let mut convs = Vec::new();
            for (cv, hc) in ds.convs.iter().zip(&hd.convs) {
                convs.push(planner.conv(hc, cv, edges[lv], cv.cin)?);
                cmax[lv] = cmax[lv].max(cv.cout);
            }
            cmax[lv] = cmax[lv].max(co);
            let tw = precision.encode(&hd.g);
            let t = &ds.transp;
            let factor = [t.stride[0], t.stride[1], t.stride[2]];
            ensure!(
                (0..3).all(|a| edges[lv][a] == edges[lv + 1][a] * factor[a]),
                "decoder {d}: transp {factor:?} maps {:?} to {:?}",
                edges[lv + 1],
                edges[lv]
            );
            decs.push(Dec { tw: gpu.upload(&tw)?, factor, tb: gpu.upload(&hd.tb)?, tcin: hd.tcin, convs, level: lv });
        }
        let head_w = gpu.upload(&host.head_w)?;
        let head_b = gpu.upload(&host.head_b)?;

        // buffers
        let enc_c: Vec<usize> = {
            let mut v = vec![0; n_levels];
            for b in &blocks {
                v[b.lout] = b.conv2.cout;
            }
            v
        };
        let mut levels = Vec::new();
        for l in 0..n_levels {
            let vox: usize = edges[l].iter().product();
            let n = vox * cmax[l];
            u32_extent("level buffer", n)?;
            let alloc = |n: usize| gpu.alloc_fill(n * precision.size(), 0xFF);
            let cat = if l + 1 < n_levels {
                u32_extent("concat buffer", vox * 2 * enc_c[l])?;
                Some(alloc(vox * 2 * enc_c[l])?)
            } else {
                None
            };
            levels.push(Level {
                dims: edges[l],
                x: alloc(n)?,
                y: alloc(n)?,
                r: alloc(n)?,
                r2: alloc(n)?,
                h: alloc(n)?,
                pa: if need_pa[l] > 0 { Some(alloc(vox * need_pa[l])?) } else { None },
                rs: if need_rs[l] { Some(alloc(n)?) } else { None },
                cat,
            });
        }
        // the decoder's GEMM output lives in `h` of the level it writes
        for d in &decs {
            ensure!(cmax[d.level] >= d.convs[0].cout.min(cmax[d.level]), "level {} too narrow", d.level);
        }
        let max_c = cmax.iter().copied().max().unwrap_or(1).max(prog.head.cin);
        ensure!(max_c <= 512, "{max_c} channels; the norm kernels take at most 512");
        let ws_bytes = planner.plans.values().map(|p| p.workspace()).max().unwrap_or(0);
        let input = gpu.alloc_fill(p.pow(3) * in_c * precision.size(), 0)?;
        let stats = [gpu.alloc(max_c * 8)?, gpu.alloc(max_c * 8)?, gpu.alloc(max_c * 8)?];
        Ok(Self {
            patch: p,
            precision,
            in_c,
            kn: Kernels::of(precision),
            stem,
            blocks,
            decs,
            head_w,
            head_b,
            head_c: host.head_cin,
            head_classes: host.head_b.len(),
            levels,
            input,
            partial: gpu.alloc(MAX_CHUNKS * max_c * 8)?,
            stats,
            ws: gpu.alloc(ws_bytes.max(1))?,
            out: (0, 'y'),
        })
    }

    /// Every distinct conv plan with its algorithm, for the report
    pub fn plans(&self) -> Vec<String> {
        let mut seen = Vec::new();
        let all = self
            .stem
            .iter()
            .chain(self.blocks.iter().flat_map(|b| [&b.conv1, &b.conv2].into_iter().chain(b.proj.iter())));
        for c in all.chain(self.decs.iter().flat_map(|d| d.convs.iter())) {
            let s = format!("{} | {}", c.plan.shape, c.plan.algo);
            if !seen.contains(&s) {
                seen.push(s);
            }
        }
        seen
    }

    /// The head of any number of classes, and the last decoder output it reads
    pub fn heads(&self) -> Head<'_> {
        let (level, which) = self.out;
        let l = &self.levels[level];
        let x = if which == 'x' { &l.x } else { &l.y };
        Head { x, cin: self.head_c, classes: self.head_classes, w: &self.head_w, b: &self.head_b }
    }

    fn conv(&self, ctx: &Ctx, c: &Conv, x: &Buf, y: &Buf) -> Result<()> {
        c.plan.run(ctx.dnn, (x.ptr(), x.bytes()), (c.w.ptr(), c.w.bytes()), (y.ptr(), y.bytes()), Some(&self.ws))
    }

    /// InstanceNorm statistics of `x`, `[rows, cols]`, after `bias`, into `self.stats[slot]`
    fn stats(
        &self,
        gpu: &Gpu,
        x: &Buf,
        bias: Option<&Buf>,
        (rows, cols): (usize, usize),
        slot: usize,
        eps: f32,
    ) -> Result<()> {
        // u3_finalize counts rows in float32, exact only up to 2^24 (a 256^3 patch)
        ensure!(rows <= 1 << 24, "InstanceNorm over {rows} rows: float32 counts are exact only to 2^24");
        let pairs = cols / 2;
        let lanes = (256 / pairs).max(1);
        let chunk_rows = rows.div_ceil(MAX_CHUNKS).max(lanes);
        let chunks = rows.div_ceil(chunk_rows);
        let bias = bias.map_or(Arg::EMPTY, |b| b.arg::<f32>());
        gpu.launch(
            self.kn.partial,
            (chunks as u32, 1, 1),
            ((pairs * lanes) as u32, 1, 1),
            0,
            &[
                self.partial.view::<[f32; 2]>(0, chunks * cols),
                pair_view(self.precision, x, rows * pairs),
                bias,
                Arg::U32(rows as u32),
                Arg::U32(cols as u32),
                Arg::U32(lanes as u32),
                Arg::U32(chunk_rows as u32),
            ],
        )?;
        gpu.launch(
            "u3_finalize",
            (cols as u32, 1, 1),
            (256, 1, 1),
            0,
            &[
                self.stats[slot].view::<[f32; 2]>(0, cols),
                self.partial.view::<[f32; 2]>(0, chunks * cols),
                Arg::U32(rows as u32),
                Arg::U32(cols as u32),
                Arg::U32(chunks as u32),
                Arg::U32(chunk_rows as u32),
                Arg::F32(eps),
            ],
        )?;
        Ok(())
    }

    /// Statistics and apply of `c`'s output `raw` into `out` (and `out2`), with an optional residual;
    /// with `pool = (pa, dims)` it also writes the `AvgPool3d(2, 2)` of the result into `pa`, of `dims` pixels
    #[allow(clippy::too_many_arguments)]
    fn norm_apply(
        &self,
        gpu: &Gpu,
        c: &Conv,
        raw: &Buf,
        vox: usize,
        residual: Residual,
        leaky: Option<f32>,
        out: (&Buf, usize, usize),
        out2: Option<(&Buf, usize, usize)>,
        pool: Option<(&Buf, [usize; 3])>,
    ) -> Result<()> {
        let n = c.norm.as_ref().expect("checked by the contract");
        let pr = self.precision;
        let cols = c.cout;
        self.stats(gpu, raw, c.bias.as_ref(), (vox, cols), 0, n.eps)?;
        let pairs = cols / 2;
        let empty_stats = Arg::EMPTY;
        let (res, rraw, rstats, rw, rb) = match residual {
            Residual::None => (Arg::EMPTY, Arg::EMPTY, empty_stats, Arg::EMPTY, Arg::EMPTY),
            Residual::Plain(r) => (pair_view(pr, r, vox * pairs), Arg::EMPTY, empty_stats, Arg::EMPTY, Arg::EMPTY),
            Residual::Normed(r, pc) => {
                let pn = pc.norm.as_ref().expect("checked by the contract");
                self.stats(gpu, r, None, (vox, cols), 1, pn.eps)?;
                (
                    Arg::EMPTY,
                    pair_view(pr, r, vox * pairs),
                    self.stats[1].view::<[f32; 2]>(0, cols),
                    pn.w.arg::<f32>(),
                    pn.b.arg::<f32>(),
                )
            }
        };
        let (o2, s2, f2) = match out2 {
            Some((b, s, f)) => (pair_arg(pr, b), s, f),
            None => (Arg::EMPTY, 0, 0),
        };
        let tail = [
            Arg::U32(cols as u32),
            Arg::U32(leaky.is_some() as u32),
            Arg::F32(leaky.unwrap_or(0.0)),
            Arg::U32(out.1 as u32),
            Arg::U32(out.2 as u32),
            Arg::U32(s2 as u32),
            Arg::U32(f2 as u32),
        ];
        let inputs = [
            pair_view(pr, raw, vox * pairs),
            c.bias.as_ref().map_or(Arg::EMPTY, |b| b.arg::<f32>()),
            self.stats[0].view::<[f32; 2]>(0, cols),
            n.w.arg::<f32>(),
            n.b.arg::<f32>(),
            res,
            rraw,
            rstats,
            rw,
            rb,
        ];
        if let Some((pa, pdims)) = pool {
            let pv: usize = pdims.iter().product();
            ensure!(8 * pv == vox, "pooling {vox} voxels into {pdims:?}");
            let args: Vec<Arg> = [pair_arg(pr, out.0), o2, pair_view(pr, pa, pv * pairs)]
                .into_iter()
                .chain(inputs)
                .chain(pdims.map(|v| Arg::U32(v as u32)))
                .chain(tail)
                .collect();
            return gpu.launch_1d(self.kn.apply_pool, pv * pairs, &args);
        }
        let args: Vec<Arg> =
            [pair_arg(pr, out.0), o2].into_iter().chain(inputs).chain([Arg::U32(vox as u32)]).chain(tail).collect();
        gpu.launch_1d(self.kn.apply, vox * pairs, &args)
    }

    /// The forward on [`Model::input`]; the result is [`Model::heads`]'s input
    pub fn forward(&mut self, ctx: &Ctx) -> Result<()> {
        let gpu = ctx.gpu;
        let lv = &self.levels;

        // stem into level 0's `x`
        let v0: usize = lv[0].dims.iter().product();
        let mut cur_in: &Buf = &self.input;
        for (i, s) in self.stem.iter().enumerate() {
            let dst = if i % 2 == 0 { &lv[0].x } else { &lv[0].y };
            self.conv(ctx, s, cur_in, &lv[0].r)?;
            self.norm_apply(gpu, s, &lv[0].r, v0, Residual::None, s.leaky, (dst, s.cout / 2, 0), None, None)?;
            cur_in = dst;
        }

        // encoder
        let mut cur_name = if self.stem.len() % 2 == 1 { 'x' } else { 'y' };
        if self.stem.is_empty() {
            cur_name = 'i';
        }
        let pick = |l: usize, n: char| -> &Buf {
            match n {
                'x' => &lv[l].x,
                'y' => &lv[l].y,
                _ => &self.input,
            }
        };
        // set when the previous block's apply already wrote this block's pooled skip input
        let mut pooled = false;
        for (bi, b) in self.blocks.iter().enumerate() {
            let input = pick(b.lin, cur_name);
            let out_name = if b.lin == b.lout && cur_name == 'x' { 'y' } else { 'x' };
            let (lo, vox) = (&lv[b.lout], lv[b.lout].dims.iter().product::<usize>());
            self.conv(ctx, &b.conv1, input, &lo.r)?;
            self.norm_apply(
                gpu,
                &b.conv1,
                &lo.r,
                vox,
                Residual::None,
                b.conv1.leaky,
                (&lo.h, b.conv1.cout / 2, 0),
                None,
                None,
            )?;
            self.conv(ctx, &b.conv2, &lo.h, &lo.r2)?;
            // skip source
            let src: &Buf = match b.pool {
                Some(_) if pooled => lo.pa.as_ref().expect("allocated for pooled skips"),
                Some(factor) => {
                    let pa = lo.pa.as_ref().expect("allocated for pooled skips");
                    let cin = b.conv1.plan.shape.c as usize;
                    let n_in = vox * factor.iter().product::<usize>();
                    let dims = lo.dims.map(|v| Arg::U32(v as u32));
                    let pr = self.precision;
                    let (out, x) = (pair_view(pr, pa, vox * cin / 2), pair_view(pr, input, n_in * cin / 2));
                    if factor == [2; 3] {
                        let args = [out, x, dims[0], dims[1], dims[2], Arg::U32(cin as u32)];
                        gpu.launch_1d(self.kn.pool2, vox * cin / 2, &args)?;
                    } else {
                        let f = factor.map(|v| Arg::U32(v as u32));
                        let args = [out, x, dims[0], dims[1], dims[2], f[0], f[1], f[2], Arg::U32(cin as u32)];
                        gpu.launch_1d(self.kn.poolk, vox * cin / 2, &args)?;
                    }
                    pa
                }
                None => input,
            };
            let residual = match &b.proj {
                Some(pc) => {
                    let rs = lo.rs.as_ref().expect("allocated for projected skips");
                    self.conv(ctx, pc, src, rs)?;
                    Residual::Normed(rs, pc)
                }
                None => Residual::Plain(src),
            };
            let last_in_level = self.blocks.get(bi + 1).is_none_or(|n| n.lout != b.lout);
            let c = b.conv2.cout;
            let out2 = if last_in_level { lo.cat.as_ref().map(|cat| (cat, c, c / 2)) } else { None };
            // the next stage's pooled skip is this output, so pool it while it is in registers
            // only a 2 x 2 x 2 pool fuses into the apply; other factors pool on their own below
            let pool = match self.blocks.get(bi + 1) {
                Some(n) if n.pool == Some([2; 3]) && n.lin == b.lout => {
                    ensure!(
                        n.conv1.plan.shape.c as usize == c,
                        "pooled skip of {c} channels into {}",
                        n.conv1.plan.shape.c
                    );
                    let pa = lv[n.lout].pa.as_ref().expect("allocated for pooled skips");
                    Some((pa, lv[n.lout].dims))
                }
                _ => None,
            };
            pooled = pool.is_some();
            self.norm_apply(
                gpu,
                &b.conv2,
                &lo.r2,
                vox,
                residual,
                Some(b.leaky),
                (pick(b.lout, out_name), c / 2, 0),
                out2,
                pool,
            )?;
            cur_name = out_name;
        }

        // decoder: lres is the bottleneck output
        let mut lres_level = self.blocks.last().map_or(0, |b| b.lout);
        let mut lres_name = cur_name;
        for dec in &self.decs {
            let l = dec.level;
            let (lo, vox) = (&lv[l], lv[l].dims.iter().product::<usize>());
            let lres = pick(lres_level, lres_name);
            let co = dec.convs[0].plan.shape.c as usize / 2;
            let vin: usize = lv[lres_level].dims.iter().product();
            let taps: usize = dec.factor.iter().product();
            let gemm = (
                (vin, taps * co, dec.tcin),
                (lres.ptr(), lres.bytes()),
                (dec.tw.ptr(), dec.tw.bytes()),
                (lo.h.ptr(), lo.h.bytes()),
            );
            match self.precision {
                Precision::Bf16 => ctx.blas.gemm_bf16(gemm.0, gemm.1, gemm.2, gemm.3)?,
                Precision::Fp16 => ctx.blas.gemm_f16(gemm.0, gemm.1, gemm.2, gemm.3)?,
                Precision::Tf32 => ctx.blas.gemm_tf32(gemm.0, gemm.1, gemm.2, gemm.3)?,
            }
            let cat = lo.cat.as_ref().expect("allocated below the bottleneck");
            let dims = lo.dims.map(|v| Arg::U32(v as u32));
            let pr = self.precision;
            let (out, g, tb) = (pair_arg(pr, cat), pair_view(pr, &lo.h, vin * taps * co / 2), dec.tb.arg::<f32>());
            let tail = [Arg::U32(co as u32), Arg::U32(co as u32), Arg::U32(0)];
            if dec.factor == [2; 3] {
                let args: Vec<Arg> = [out, g, tb, dims[0], dims[1], dims[2]].into_iter().chain(tail).collect();
                gpu.launch_1d(self.kn.up, vox * co / 2, &args)?;
            } else {
                let f = dec.factor.map(|v| Arg::U32(v as u32));
                let args: Vec<Arg> =
                    [out, g, tb, dims[0], dims[1], dims[2], f[0], f[1], f[2]].into_iter().chain(tail).collect();
                gpu.launch_1d(self.kn.upk, vox * co / 2, &args)?;
            }
            let mut src: &Buf = cat;
            for (i, cv) in dec.convs.iter().enumerate() {
                let dst = if dec_dst(i) == 'y' { &lo.y } else { &lo.x };
                self.conv(ctx, cv, src, &lo.r)?;
                self.norm_apply(gpu, cv, &lo.r, vox, Residual::None, cv.leaky, (dst, cv.cout / 2, 0), None, None)?;
                src = dst;
            }
            lres_level = l;
            lres_name = dec_dst(dec.convs.len() - 1);
        }
        ensure!(lres_level == 0, "the last decoder stage must end at level 0, not {lres_level}");
        self.out = (0, lres_name);
        Ok(())
    }
}

/// A block's residual
enum Residual<'a> {
    None,
    /// A tensor of the output's shape in the storage type
    Plain(&'a Buf),
    /// A bias-free conv output, normalised on load by that conv's norm
    Normed(&'a Buf, &'a Conv),
}

/// `len` channel pairs of `b` from its start, as the glue kernels of `precision` take them: `u32` pairs of 16-bit
/// values, or `[f32; 2]`
fn pair_view(precision: Precision, b: &Buf, len: usize) -> Arg {
    match precision {
        Precision::Bf16 | Precision::Fp16 => b.view::<u32>(0, len),
        Precision::Tf32 => b.view::<[f32; 2]>(0, len),
    }
}

/// All of `b` as channel pairs of `precision`'s storage, as [`pair_view`]
fn pair_arg(precision: Precision, b: &Buf) -> Arg {
    match precision {
        Precision::Bf16 | Precision::Fp16 => b.arg::<u32>(),
        Precision::Tf32 => b.arg::<[f32; 2]>(),
    }
}

/// Fails unless `n` elements index with u32
pub fn u32_extent(what: &str, n: usize) -> Result<()> {
    u32::try_from(n).map(|_| ()).wrap_err_with(|| format!("{what}: {n} elements overflow u32 indexing"))
}

/// The buffer (`'y'` or `'x'`) decoder conv `i` of a stage writes: they alternate from `y`, so a
/// stage of n convs ends in `dec_dst(n - 1)`, which is where the head must read the last stage
fn dec_dst(i: usize) -> char {
    if i.is_multiple_of(2) { 'y' } else { 'x' }
}

#[cfg(test)]
mod dec_tests {
    use super::dec_dst;

    /// A last stage of one or three convs ends in y and of two in x, and the head reads what the
    /// forward recorded, so an even-conv last stage (which the exporter accepts) reads its output
    #[test]
    fn decoder_buffers_alternate_from_y() {
        assert_eq!([0, 1, 2].map(dec_dst), ['y', 'x', 'y']);
    }
}
