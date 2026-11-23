//! The 3D ResEnc U-Net forward on Metal at batch 1, in float32, built from an export's layer program
//!
//! Every resolution level owns three activation buffers in the zero-padded channels-last layout of `common.metal`,
//! all of that level's width (the program keeps one channel count per level). A block reads one, writes its conv1
//! output into the second and its conv2 output into the third, and the next block rotates them; InstanceNorm,
//! the residual and LeakyReLU run in place on the conv output. The last block of a level leaves its output where it
//! is, and that buffer is the level's skip: the decoder at that level runs its transposed conv into one of the other
//! two and its first conv over `(up, skip)` as one accumulation with the weight split on the host, so no concat
//! buffer exists. Where a block pools its skip, the pooled input and the 1x1 projection's output are plain
//! `[voxels, c]` buffers of the block's level

use color_eyre::eyre::{Result, ensure};

use std::cell::Cell;

use super::gpu::{Arg, Buf, Gpu};
use super::winograd;
use crate::ensure_supported;
use crate::unet3d::host::{HostConv, HostNet};
use crate::unet3d::{ConvSpec, Precision, Program};

/// Upper bound on InstanceNorm partial-sum chunks per tensor, as on CUDA
const MAX_CHUNKS: usize = 4096;
/// Rows a tap conv tile may read past a padded buffer's last row (a 64-row tile and two lines of shifts), allocated
/// and zeroed so every view stays inside its buffer
const SLACK_ROWS: usize = 64 + 8;

/// One level's dims (z, y, x) and the rows its padded layout takes, slack included
pub(crate) fn padded_rows(d: [usize; 3]) -> usize {
    (d[0] + 2) * (d[1] + 2) * (d[2] + 2) + SLACK_ROWS + 2 * (d[2] + 2)
}

/// An InstanceNorm's float32 affine and eps
struct Norm {
    w: Buf,
    b: Buf,
    eps: f32,
}

/// A conv's weights on the GPU, for the tap convs or for Winograd
enum Weights {
    /// Channels-last `[cout, taps, cin]` float32; for the decoder's first conv over `(up, skip)`, `w` is the up half
    /// `[cout, 27, cin / 2]` and `skip` the skip half
    Direct { w: Buf, skip: Option<Buf> },
    /// Winograd F(4,3)'s transformed weights `[216, cout, cin]`, over the whole concatenated input
    Winograd(Buf),
}

/// A conv's weights, its bias and the norm after it
struct Conv {
    weights: Weights,
    bias: Option<Buf>,
    norm: Option<Norm>,
    leaky: Option<f32>,
    cin: usize,
    cout: usize,
    stride: [usize; 3],
}

impl Conv {
    /// The conv's weights uploaded for the tap convs, split into `(up, skip)` halves when `split`, or transformed for
    /// Winograd when `winograd`
    fn new(gpu: &Gpu, host: &HostConv, spec: &ConvSpec, split: bool, winograd: bool) -> Result<Self> {
        let taps: usize = spec.kernel.iter().product();
        let (k, c) = (host.cout, host.cin);
        let weights = if winograd {
            ensure!(spec.kernel == [3, 3, 3] && spec.stride == [1, 1, 1], "Winograd on a {:?} conv", spec.kernel);
            Weights::Winograd(gpu.upload(&winograd::transform_weights(&host.w, k, c))?)
        } else if split {
            ensure!(c.is_multiple_of(2), "a concat conv of {c} input channels");
            let half = c / 2;
            let (mut up, mut skip) = (Vec::with_capacity(k * taps * half), Vec::with_capacity(k * taps * half));
            for row in host.w.chunks_exact(c) {
                up.extend_from_slice(&row[..half]);
                skip.extend_from_slice(&row[half..]);
            }
            Weights::Direct { w: gpu.upload(&up)?, skip: Some(gpu.upload(&skip)?) }
        } else {
            Weights::Direct { w: gpu.upload(&host.w)?, skip: None }
        };
        let bias = host.bias.as_ref().map(|b| gpu.upload(b)).transpose()?;
        let norm = match &host.norm {
            Some((w, b, eps)) => Some(Norm { w: gpu.upload(w)?, b: gpu.upload(b)?, eps: *eps }),
            None => None,
        };
        let stride = [spec.stride[0], spec.stride[1], spec.stride[2]];
        Ok(Self { weights, bias, norm, leaky: spec.leaky, cin: c, cout: k, stride })
    }

    /// The tap convs' weights and the concat conv's skip half
    fn direct(&self) -> Result<(&Buf, Option<&Buf>)> {
        match &self.weights {
            Weights::Direct { w, skip } => Ok((w, skip.as_ref())),
            Weights::Winograd(_) => color_eyre::eyre::bail!("a Winograd conv has no tap-conv weights"),
        }
    }

    fn bytes(&self) -> usize {
        let w = match &self.weights {
            Weights::Direct { w, skip } => w.bytes() + skip.as_ref().map_or(0, Buf::bytes),
            Weights::Winograd(u) => u.bytes(),
        };
        w + self.bias.as_ref().map_or(0, Buf::bytes) + self.norm.as_ref().map_or(0, |n| n.w.bytes() + n.b.bytes())
    }
}

/// The levels whose stride-1 3x3x3 convs run Winograd: `VOLKIT_METAL_WINOGRAD`, a comma-separated level list or
/// `none`, defaulting to 0, 1 and 2. Level 0 takes its scratch from level 1, so it needs a second level. Level 3
/// stays direct by default: its transformed weights would take about 680 MB
fn winograd_levels() -> Result<Vec<usize>> {
    let spec = std::env::var("VOLKIT_METAL_WINOGRAD").unwrap_or_else(|_| "0,1,2".into());
    if spec == "none" {
        return Ok(Vec::new());
    }
    spec.split(',')
        .map(|v| {
            v.trim().parse().map_err(|_| color_eyre::eyre::eyre!("VOLKIT_METAL_WINOGRAD={spec:?}: levels or none"))
        })
        .collect()
}

/// The most tiles one level-0 Winograd chunk takes: `VOLKIT_METAL_WINO_TILES`, 0 for as many as the scratch holds,
/// default 256. Level 0's chunks of 256 tiles keep V and M (7 MB each) in the system-level cache. The wider levels
/// run uncapped: each chunk re-reads all of U (up to 14 MB on level 2), and an unpaired sweep measured caps of 64 to
/// 1024 tiles there about 20 ms a tile slower
fn winograd_tiles() -> Result<usize> {
    match std::env::var("VOLKIT_METAL_WINO_TILES") {
        Ok(v) => v.trim().parse().map_err(|_| color_eyre::eyre::eyre!("VOLKIT_METAL_WINO_TILES={v:?}: a tile count")),
        Err(_) => Ok(256),
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
    lout: usize,
}

/// One decoder stage, writing level `level`
struct Dec {
    /// The transposed conv as a GEMM weight `[taps * cout, cin]`, row `tap * cout + co`
    tw: Buf,
    tb: Buf,
    factor: [usize; 3],
    tcin: usize,
    tcout: usize,
    convs: Vec<Conv>,
    level: usize,
}

/// One resolution level's buffers
struct Level {
    dims: [usize; 3],
    /// Channels
    c: usize,
    rows: usize,
    bufs: [Buf; 3],
    /// The pooled skip input `[voxels, c_in]` and the skip projection `[voxels, c]`, where a block needs them
    pa: Option<Buf>,
    rs: Option<Buf>,
}

/// A block's residual
enum Residual<'a> {
    None,
    /// A normalised activation of the output's level, padded
    Plain(&'a Buf),
    /// A bias-free 1x1 conv output, plain, normalised on load by that conv's norm
    Normed(&'a Buf, &'a Conv),
    /// An average-pooled skip, plain, where the stage strides without changing width
    Pooled(&'a Buf),
}

/// The head's input and weights, for the engine's fold
pub(crate) struct Head<'a> {
    /// The last decoder output, padded `[p + 2]^3 x cin`
    pub x: &'a Buf,
    pub cin: usize,
    /// The weight `[classes, cin]` and bias
    pub w: &'a Buf,
    pub b: &'a Buf,
}

/// The network on the GPU, at batch 1
pub(crate) struct Model {
    pub patch: usize,
    /// The levels whose stride-1 convs run Winograd, ascending
    pub winograd: Vec<usize>,
    /// The most tiles a level-0 Winograd chunk takes, 0 for as many as the scratch holds
    wino_tiles: usize,
    stem: Conv,
    blocks: Vec<Block>,
    decs: Vec<Dec>,
    head_w: Buf,
    head_b: Buf,
    head_c: usize,
    levels: Vec<Level>,
    part: Buf,
    stats: [Buf; 2],
    /// Bound to pointer parameters a dispatch does not use
    dummy: Buf,
    /// (level 0, buffer) of the last decoder output
    out: usize,
    /// The level-0 buffer Winograd convs use as scratch during a forward (one that is free from the end of level 0's
    /// encoder to its decoder), and whether a conv used it, so its padding is zeroed again before level 0 reads it
    scratch: Cell<usize>,
    scratch_used: Cell<bool>,
}

/// Dispatches `in_partial` over the `[dims, c]` activation `x` (padded layout, or plain) into `part`, whose `[chunks,
/// c]` partials `in_finalize` then merges; returns `chunks`, at most [`MAX_CHUNKS`]. A thread group takes a run of
/// whole lines and up to 64 channel quads, split evenly when there are more
pub(crate) fn in_partial(gpu: &Gpu, part: &Buf, x: &Buf, dims: [usize; 3], c: usize, padded: bool) -> Result<usize> {
    ensure!(c.is_multiple_of(4), "InstanceNorm statistics over {c} channels: in_partial takes channel quads");
    let quads = c / 4;
    let qpg = quads.div_ceil(quads.div_ceil(64));
    let lanes = 256 / qpg;
    let lines = dims[0] * dims[1];
    let chunk_lines = lines.div_ceil(MAX_CHUNKS);
    let chunks = lines.div_ceil(chunk_lines);
    ensure!(part.bytes() >= chunks * c * 16, "in_partial: {} bytes for {chunks} x {c} partials", part.bytes());
    let args = [
        part.arg(),
        x.arg(),
        dims[0].into(),
        dims[1].into(),
        dims[2].into(),
        c.into(),
        usize::from(padded).into(),
        chunk_lines.into(),
        qpg.into(),
        lanes.into(),
    ];
    gpu.dispatch("in_partial", [chunks, quads.div_ceil(qpg), 1], [qpg * lanes, 1, 1], &args)?;
    Ok(chunks)
}

/// The tap conv tile for `o` output channels: 64 wide unless that wastes half the tile
fn bn(o: usize) -> usize {
    if o >= 64 { 64 } else { 32 }
}

fn taps_s1_name(bm: usize, bn: usize) -> &'static str {
    match (bm, bn) {
        (64, 64) => "taps_s1_64x64",
        (64, 32) => "taps_s1_64x32",
        (32, 64) => "taps_s1_32x64",
        _ => "taps_s1_32x32",
    }
}

fn taps_sn_name(bm: usize, bn: usize) -> &'static str {
    match (bm, bn) {
        (64, 64) => "taps_sn_64x64",
        (64, 32) => "taps_sn_64x32",
        (32, 64) => "taps_sn_32x64",
        _ => "taps_sn_32x32",
    }
}

fn gemm_name(bn: usize) -> &'static str {
    if bn == 64 { "gemm_nt_64x64" } else { "gemm_nt_64x32" }
}

fn tconv_name(bn: usize) -> &'static str {
    if bn == 64 { "tconv_64x64" } else { "tconv_64x32" }
}

impl Model {
    /// Uploads `host`'s weights for `prog` (cubic patch of edge `p`) and allocates the buffers. Only float32 runs on
    /// Metal; the stem's first conv must take the one CT channel
    pub fn new(gpu: &Gpu, host: &HostNet, prog: &Program, p: usize) -> Result<Self> {
        prog.check_axes()?;
        ensure_supported!(host.precision == Precision::Tf32, "the Metal forward runs float32 only");
        ensure_supported!(
            prog.stem.len() == 1 && prog.stem[0].cin == 1 && prog.stem[0].kernel == [3, 3, 3],
            "the Metal forward's stem is one 3x3x3 conv on one channel"
        );
        ensure_supported!(prog.stem[0].cout.is_multiple_of(8), "stem of {} channels", prog.stem[0].cout);
        let stem = Conv::new(gpu, &host.stem[0], &prog.stem[0], false, false)?;
        let wino = winograd_levels()?;

        let mut blocks = Vec::new();
        let (mut dims, mut level) = ([p; 3], 0usize);
        let mut edges = vec![dims];
        let mut widths = vec![stem.cout];
        let mut need_pa = vec![0usize];
        let mut need_rs = vec![false];
        for (b, hb) in prog.stages.iter().flatten().zip(&host.blocks) {
            let stride = [b.conv1.stride[0], b.conv1.stride[1], b.conv1.stride[2]];
            if stride != [1; 3] {
                dims = std::array::from_fn(|a| dims[a] / stride[a]);
                level += 1;
                edges.push(dims);
                widths.push(b.conv2.cout);
                need_pa.push(0);
                need_rs.push(false);
            }
            let lout = level;
            ensure_supported!(b.conv1.kernel == [3, 3, 3] && b.conv2.kernel == [3, 3, 3], "block convs must be 3x3x3");
            ensure_supported!(
                widths[lout] == b.conv1.cout && widths[lout] == b.conv2.cout,
                "the Metal forward keeps one width per level; block at level {lout} is {} -> {}",
                b.conv1.cout,
                b.conv2.cout
            );
            let at = wino.contains(&lout) && winograd::fits(dims);
            let conv1 = Conv::new(gpu, &hb.conv1, &b.conv1, false, at && stride == [1; 3])?;
            let conv2 = Conv::new(gpu, &hb.conv2, &b.conv2, false, at)?;
            let proj = match (&b.skip.proj, &hb.proj) {
                (Some(pj), Some(hp)) => Some(Conv::new(gpu, hp, pj, false, false)?),
                _ => None,
            };
            let pool = Some(b.skip.pool.axes()).filter(|&a| a != [1; 3]);
            ensure_supported!(
                proj.is_none() || pool.is_some(),
                "a projected skip without a pool (a width change at stride 1)"
            );
            if pool.is_some() {
                need_pa[lout] = need_pa[lout].max(b.conv1.cin);
            }
            need_rs[lout] |= proj.is_some();
            blocks.push(Block { conv1, conv2, pool, proj, leaky: b.leaky, lout });
        }
        let n_levels = level + 1;
        ensure!(prog.decoder.len() + 1 == n_levels, "{} decoder stages for {n_levels} levels", prog.decoder.len());

        let mut decs = Vec::new();
        for (d, (ds, hd)) in prog.decoder.iter().zip(&host.decs).enumerate() {
            let lv = n_levels - 2 - d;
            let t = &ds.transp;
            ensure_supported!(t.cout == widths[lv], "decoder {d}: transp to {}, level {lv} is {}", t.cout, widths[lv]);
            let mut convs = Vec::new();
            for (i, (cv, hc)) in ds.convs.iter().zip(&hd.convs).enumerate() {
                ensure_supported!(cv.kernel == [3, 3, 3] && cv.cout == widths[lv], "decoder {d} conv {i}");
                let at = wino.contains(&lv) && winograd::fits(edges[lv]);
                convs.push(Conv::new(gpu, hc, cv, i == 0 && !at, at)?);
            }
            let taps: usize = t.kernel.iter().product();
            let (ci, co) = (hd.tcin, t.cout);
            // the host's `[cin, taps * cout]` transposed, so a row per output column
            let mut tw = vec![0f32; ci * taps * co];
            for i in 0..ci {
                for n in 0..taps * co {
                    tw[n * ci + i] = hd.g[i * taps * co + n];
                }
            }
            let factor = [t.stride[0], t.stride[1], t.stride[2]];
            decs.push(Dec {
                tw: gpu.upload(&tw)?,
                tb: gpu.upload(&hd.tb)?,
                factor,
                tcin: ci,
                tcout: co,
                convs,
                level: lv,
            });
        }
        let head_w = gpu.upload(&host.head_w)?;
        let head_b = gpu.upload(&host.head_b)?;

        let mut levels = Vec::new();
        for l in 0..n_levels {
            let (d, c) = (edges[l], widths[l]);
            ensure!(c.is_multiple_of(4), "level {l} of {c} channels; the norm kernels take multiples of 4");
            let rows = padded_rows(d);
            let vox: usize = d.iter().product();
            ensure!(u32::try_from(rows * c).is_ok(), "level {l}: {rows} x {c} overflows 32-bit indexing");
            let alloc = || gpu.alloc(rows * c * 4);
            levels.push(Level {
                dims: d,
                c,
                rows,
                bufs: [alloc()?, alloc()?, alloc()?],
                pa: if need_pa[l] > 0 { Some(gpu.alloc(vox * need_pa[l] * 4)?) } else { None },
                rs: if need_rs[l] { Some(gpu.alloc(vox * c * 4)?) } else { None },
            });
        }
        let max_c = widths.iter().copied().max().unwrap_or(1);
        let mut winograd: Vec<usize> = wino.into_iter().filter(|&l| l < n_levels && winograd::fits(edges[l])).collect();
        winograd.sort_unstable();
        winograd.dedup();
        ensure!(
            !winograd.contains(&0) || n_levels > 1,
            "Winograd on level 0 takes its scratch from level 1, and this network has one level"
        );
        Ok(Self {
            patch: p,
            winograd,
            wino_tiles: winograd_tiles()?,
            stem,
            blocks,
            decs,
            head_w,
            head_b,
            head_c: host.head_cin,
            levels,
            part: gpu.alloc(MAX_CHUNKS * max_c * 16)?,
            stats: [gpu.alloc(max_c * 8)?, gpu.alloc(max_c * 8)?],
            dummy: gpu.alloc(16)?,
            out: 0,
            scratch: Cell::new(usize::MAX),
            scratch_used: Cell::new(false),
        })
    }

    /// The head and the last decoder output it reads
    pub fn head(&self) -> Head<'_> {
        Head { x: &self.levels[0].bufs[self.out], cin: self.head_c, w: &self.head_w, b: &self.head_b }
    }

    /// Device bytes the model's buffers hold
    pub fn bytes(&self) -> usize {
        let conv = Conv::bytes;
        let mut total = conv(&self.stem) + self.head_w.bytes() + self.head_b.bytes();
        for b in &self.blocks {
            total += conv(&b.conv1) + conv(&b.conv2) + b.proj.as_ref().map_or(0, conv);
        }
        for d in &self.decs {
            total += d.tw.bytes() + d.tb.bytes() + d.convs.iter().map(conv).sum::<usize>();
        }
        for l in &self.levels {
            total += l.bufs.iter().map(Buf::bytes).sum::<usize>()
                + l.pa.as_ref().map_or(0, Buf::bytes)
                + l.rs.as_ref().map_or(0, Buf::bytes);
        }
        total + self.part.bytes()
    }

    /// InstanceNorm statistics of the `[dims, c]` activation `x` into `stats[0]` when it is a padded conv output, or
    /// `stats[1]` when it is a plain skip projection, the residual normalised in the same apply
    fn stats(&self, gpu: &Gpu, x: &Buf, dims: [usize; 3], c: usize, padded: bool, eps: f32) -> Result<()> {
        let slot = usize::from(!padded);
        let rows: usize = dims.iter().product();
        // in_finalize counts rows in float32, exact only up to 2^24 (a 256^3 patch)
        ensure!(rows <= 1 << 24, "InstanceNorm over {rows} rows: float32 counts are exact only to 2^24");
        let chunks = in_partial(gpu, &self.part, x, dims, c, padded)?;
        gpu.dispatch(
            "in_finalize",
            [c, 1, 1],
            [256, 1, 1],
            &[self.stats[slot].arg(), self.part.arg(), chunks.into(), c.into(), eps.into()],
        )
    }

    /// The norm, residual and activation after conv `c`, in place on its padded output `y` at level `l`
    fn norm_apply(&self, gpu: &Gpu, l: usize, y: &Buf, c: &Conv, residual: Residual, leaky: Option<f32>) -> Result<()> {
        let lv = &self.levels[l];
        let n = c.norm.as_ref().ok_or_else(|| color_eyre::eyre::eyre!("conv without a norm"))?;
        self.stats(gpu, y, lv.dims, c.cout, true, n.eps)?;
        let d = &self.dummy;
        let (mode, res, rstats, rw, rb) = match residual {
            Residual::None => (0u32, d, d, d, d),
            Residual::Plain(r) => (1, r, d, d, d),
            Residual::Pooled(r) => (3, r, d, d, d),
            Residual::Normed(r, pc) => {
                let pn = pc.norm.as_ref().ok_or_else(|| color_eyre::eyre::eyre!("skip projection without a norm"))?;
                self.stats(gpu, r, lv.dims, pc.cout, false, pn.eps)?;
                (2, r, &self.stats[1], &pn.w, &pn.b)
            }
        };
        let vox: usize = lv.dims.iter().product();
        let dims = lv.dims.map(Arg::from);
        gpu.dispatch_1d(
            "in_apply",
            vox * c.cout / 4,
            &[
                y.arg(),
                self.stats[0].arg(),
                n.w.arg(),
                n.b.arg(),
                res.arg(),
                rstats.arg(),
                rw.arg(),
                rb.arg(),
                dims[0],
                dims[1],
                dims[2],
                c.cout.into(),
                Arg::U32(mode),
                Arg::U32(u32::from(leaky.is_some())),
                Arg::F32(leaky.unwrap_or(0.0)),
            ],
        )
    }

    /// A stride-1 3x3x3 conv of level `l`'s padded `x` (and, for the decoder's concat, `x2`) into `y`
    fn conv_s1(&self, gpu: &Gpu, l: usize, c: &Conv, x: &Buf, x2: Option<&Buf>, y: &Buf) -> Result<()> {
        let lv = &self.levels[l];
        if let Weights::Winograd(u) = &c.weights {
            let inputs = match x2 {
                Some(x2) => vec![(x, c.cin / 2), (x2, c.cin / 2)],
                None => vec![(x, c.cin)],
            };
            let out = (y, c.cout, c.bias.as_ref());
            if l == 0 {
                // level 1 holds nothing live while level 0 runs (its encoder before level 1's, its decoder after), so
                // a level-1 buffer is the scratch, its padding zeroed again at once for level 1's convs
                let l1 = &self.levels[1];
                winograd::conv(gpu, u, &inputs, out, lv.dims, &l1.bufs[0], self.wino_tiles)?;
                return winograd::zero_border(gpu, &l1.bufs[0], l1.dims, l1.c, l1.rows);
            }
            let k = self.scratch.get();
            ensure!(k < 3, "a Winograd conv at level {l} before level 0's scratch is free");
            self.scratch_used.set(true);
            return winograd::conv(gpu, u, &inputs, out, lv.dims, &self.levels[0].bufs[k], 0);
        }
        let [d, h, w] = lv.dims;
        let span = h * (w + 2);
        // the row tile that computes fewer discarded rows, 64 on a tie
        let waste = |bm: usize| span.div_ceil(bm) * bm - span;
        let bm = if waste(32) < waste(64) { 32 } else { 64 };
        let bnn = bn(c.cout);
        let (w1, w_skip) = c.direct()?;
        let (cin, c2) = match (x2, w_skip) {
            (Some(_), Some(_)) => (c.cin / 2, c.cin / 2),
            (None, None) => (c.cin, 0),
            _ => color_eyre::eyre::bail!("a concat conv needs both inputs and a split weight"),
        };
        let w2 = w_skip.unwrap_or(&self.dummy);
        let bias = c.bias.as_ref().unwrap_or(&self.dummy);
        gpu.dispatch(
            taps_s1_name(bm, bnn),
            [c.cout.div_ceil(bnn), d * span.div_ceil(bm), 1],
            [128, 1, 1],
            &[
                x.arg(),
                w1.arg(),
                x2.unwrap_or(&self.dummy).arg(),
                w2.arg(),
                y.arg(),
                bias.arg(),
                d.into(),
                h.into(),
                w.into(),
                cin.into(),
                c2.into(),
                c.cout.into(),
                lv.rows.into(),
                Arg::U32(u32::from(c.bias.is_some())),
            ],
        )
    }

    /// A strided 3x3x3 conv from level `lin`'s padded `x` into level `lout`'s padded `y`
    fn conv_sn(&self, gpu: &Gpu, lin: usize, lout: usize, c: &Conv, x: &Buf, y: &Buf) -> Result<()> {
        let (di, dout) = (self.levels[lin].dims, self.levels[lout].dims);
        let bm = if dout[2] >= 64 { 64 } else { 32 };
        let bnn = bn(c.cout);
        let bias = c.bias.as_ref().unwrap_or(&self.dummy);
        let s = c.stride.map(Arg::from);
        gpu.dispatch(
            taps_sn_name(bm, bnn),
            [c.cout.div_ceil(bnn), dout[0] * dout[1] * dout[2].div_ceil(bm), 1],
            [128, 1, 1],
            &[
                x.arg(),
                c.direct()?.0.arg(),
                y.arg(),
                bias.arg(),
                di[0].into(),
                di[1].into(),
                di[2].into(),
                c.cin.into(),
                dout[0].into(),
                dout[1].into(),
                dout[2].into(),
                c.cout.into(),
                s[0],
                s[1],
                s[2],
                Arg::U32(u32::from(c.bias.is_some())),
            ],
        )
    }

    /// The forward of the uploaded tile `tile` (`p^3` uint8) with its normalisation `table` (256 float32); the result
    /// is [`Model::head`]'s input
    pub fn forward(&mut self, gpu: &Gpu, tile: Arg, table: Arg) -> Result<()> {
        let p = self.patch;
        let lv = &self.levels;

        // stem into level 0's buffer 0
        let s = &self.stem;
        gpu.dispatch_1d(
            "stem",
            p.pow(3) * s.cout / 8,
            &[
                lv[0].bufs[0].arg(),
                tile,
                table,
                s.direct()?.0.arg(),
                s.bias.as_ref().unwrap_or(&self.dummy).arg(),
                p.into(),
                s.cout.into(),
                (p.pow(3) * s.cout / 8).into(),
            ],
        )?;
        self.norm_apply(gpu, 0, &lv[0].bufs[0], s, Residual::None, s.leaky)?;

        // encoder: (level, buffer) of the current activation, and each level's skip buffer
        let mut cur = (0usize, 0usize);
        let mut skip = vec![usize::MAX; lv.len()];
        for (bi, b) in self.blocks.iter().enumerate() {
            let (li, xi) = cur;
            let lo = b.lout;
            let free: Vec<usize> = (0..3).filter(|&k| !(lo == li && k == xi)).collect();
            let (hi, ri) = (free[0], free[1]);
            let (input, h, r) = (&lv[li].bufs[xi], &lv[lo].bufs[hi], &lv[lo].bufs[ri]);
            if b.conv1.stride == [1; 3] {
                self.conv_s1(gpu, lo, &b.conv1, input, None, h)?;
            } else {
                self.conv_sn(gpu, li, lo, &b.conv1, input, h)?;
            }
            self.norm_apply(gpu, lo, h, &b.conv1, Residual::None, b.conv1.leaky)?;
            self.conv_s1(gpu, lo, &b.conv2, h, None, r)?;
            // a strided block's skip average-pools its input into the plain `pa` first
            let pooled = match b.pool {
                Some(f) => {
                    let pa = lv[lo].pa.as_ref().ok_or_else(|| color_eyre::eyre::eyre!("no pooled skip buffer"))?;
                    let d = lv[lo].dims;
                    let cin = b.conv1.cin;
                    let fa = f.map(Arg::from);
                    let args =
                        [pa.arg(), input.arg(), d[0].into(), d[1].into(), d[2].into(), cin.into(), fa[0], fa[1], fa[2]];
                    gpu.dispatch_1d("avgpool", d.iter().product::<usize>() * cin, &args)?;
                    Some(pa)
                }
                None => None,
            };
            let residual = match (&b.proj, pooled) {
                (Some(pc), Some(src)) => {
                    let rs = lv[lo].rs.as_ref().ok_or_else(|| color_eyre::eyre::eyre!("no skip projection buffer"))?;
                    let m: usize = lv[lo].dims.iter().product();
                    let bnn = bn(pc.cout);
                    gpu.dispatch(
                        gemm_name(bnn),
                        [pc.cout.div_ceil(bnn), m.div_ceil(64), 1],
                        [128, 1, 1],
                        &[src.arg(), pc.direct()?.0.arg(), rs.arg(), m.into(), pc.cout.into(), pc.cin.into()],
                    )?;
                    Residual::Normed(rs, pc)
                }
                (None, Some(src)) => Residual::Pooled(src),
                (Some(_), None) => color_eyre::eyre::bail!("a projected skip without a pool reads a padded input"),
                (None, None) => {
                    ensure!(li == lo, "an identity skip across levels");
                    Residual::Plain(input)
                }
            };
            self.norm_apply(gpu, lo, r, &b.conv2, residual, Some(b.leaky))?;
            cur = (lo, ri);
            let last_in_level = self.blocks.get(bi + 1).is_none_or(|n| n.lout != lo);
            if last_in_level {
                skip[lo] = ri;
                if lo == 0 {
                    // level 0's other two buffers stay free until its decoder: the first is Winograd's scratch
                    self.scratch.set((0..3).find(|&k| k != ri).unwrap_or(0));
                    self.scratch_used.set(false);
                }
            }
        }

        // decoder
        for dec in &self.decs {
            let l = dec.level;
            if l == 0 && self.scratch_used.get() {
                // the scratch is the buffer this stage's transposed conv writes, and its padding must be zero again
                let l0 = &lv[0];
                winograd::zero_border(gpu, &l0.bufs[self.scratch.get()], l0.dims, l0.c, l0.rows)?;
                self.scratch_used.set(false);
            }
            let (li, xi) = cur;
            ensure!(li == l + 1, "decoder at level {l} reads level {li}");
            let si = skip[l];
            let free: Vec<usize> = (0..3).filter(|&k| k != si).collect();
            let (ui, oi) = (free[0], free[1]);
            let (x, up) = (&lv[li].bufs[xi], &lv[l].bufs[ui]);
            let di = lv[li].dims;
            let n = dec.factor.iter().product::<usize>() * dec.tcout;
            let bnn = bn(n);
            let f = dec.factor.map(Arg::from);
            gpu.dispatch(
                tconv_name(bnn),
                [n.div_ceil(bnn), ((di[0] + 2) * (di[1] + 2) * (di[2] + 2)).div_ceil(64), 1],
                [128, 1, 1],
                &[
                    x.arg(),
                    dec.tw.arg(),
                    up.arg(),
                    dec.tb.arg(),
                    di[0].into(),
                    di[1].into(),
                    di[2].into(),
                    dec.tcin.into(),
                    dec.tcout.into(),
                    f[0],
                    f[1],
                    f[2],
                ],
            )?;
            let mut src = (ui, Some(si));
            let mut dst = oi;
            for cv in &dec.convs {
                let x2 = src.1.map(|k| &lv[l].bufs[k]);
                self.conv_s1(gpu, l, cv, &lv[l].bufs[src.0], x2, &lv[l].bufs[dst])?;
                self.norm_apply(gpu, l, &lv[l].bufs[dst], cv, Residual::None, cv.leaky)?;
                let next = (0..3).find(|&k| k != dst && k != si).unwrap_or(ui);
                src = (dst, None);
                dst = next;
            }
            cur = (l, src.0);
        }
        ensure!(cur.0 == 0, "the last decoder stage must end at level 0, not {}", cur.0);
        self.out = cur.1;
        Ok(())
    }
}
