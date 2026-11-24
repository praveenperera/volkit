//! Winograd F(4x4x4, 3x3x3) for the stride-1 3x3x3 convs of levels 0 to 2 (kernels in `winograd.metal`): the
//! weights are transformed once at load, in float64 then rounded to float32, and each conv runs as an input
//! transform, 216 batched GEMMs and an output transform, over chunks of tiles sized to the scratch buffer it is
//! given

use color_eyre::eyre::{Result, ensure};

use super::gpu::{Arg, Buf, Gpu};

/// Transform points per tile, 6^3
pub(crate) const POINTS: usize = 216;

/// F(4, 3)'s weight transform G, 6 x 3, points 0, -1, 1, 1/2, -1/2 and infinity scaled as Lavin and Gray's
const G: [[f64; 3]; 6] = [
    [1.0 / 4.0, 0.0, 0.0],
    [-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0],
    [-1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0],
    [1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0],
    [1.0 / 24.0, -1.0 / 12.0, 1.0 / 6.0],
    [0.0, 0.0, 1.0],
];

/// The transformed weights `[216, o, c]` of a 3x3x3 conv's channels-last weights `w` `[o, 27, c]` (tap
/// `(dz * 3 + dy) * 3 + dx`): `U = (G x G x G) g` per (output, input) channel pair, point `(pz * 6 + py) * 6 + px`
pub(crate) fn transform_weights(w: &[f32], o: usize, c: usize) -> Vec<f32> {
    assert_eq!(w.len(), o * 27 * c, "weights of {} values for {o} x 27 x {c}", w.len());
    let mut u = vec![0f32; POINTS * o * c];
    let mut g = [0f64; 27];
    for k in 0..o {
        for ci in 0..c {
            for (t, v) in g.iter_mut().enumerate() {
                *v = f64::from(w[(k * 27 + t) * c + ci]);
            }
            // separable: x, then y, then z, each 3 -> 6
            let mut gx = [0f64; 3 * 3 * 6];
            for zy in 0..9 {
                for (px, row) in G.iter().enumerate() {
                    gx[zy * 6 + px] = (0..3).map(|e| row[e] * g[zy * 3 + e]).sum();
                }
            }
            let mut gy = [0f64; 3 * 6 * 6];
            for z in 0..3 {
                for (py, row) in G.iter().enumerate() {
                    for px in 0..6 {
                        gy[(z * 6 + py) * 6 + px] = (0..3).map(|b| row[b] * gx[(z * 3 + b) * 6 + px]).sum();
                    }
                }
            }
            for (pz, row) in G.iter().enumerate() {
                for pyx in 0..36 {
                    let v: f64 = (0..3).map(|a| row[a] * gy[a * 36 + pyx]).sum();
                    u[((pz * 36 + pyx) * o + k) * c + ci] = v as f32;
                }
            }
        }
    }
    u
}

/// Whether a level of interior `dims` can run Winograd: every axis a multiple of the 4-voxel output tile
pub(crate) fn fits(dims: [usize; 3]) -> bool {
    dims.iter().all(|&v| v > 0 && v.is_multiple_of(4))
}

/// How many tiles of a conv with `cv` input and `o` output channels a scratch buffer of `bytes` holds: each tile's V
/// and M take 216 points of every channel
pub(crate) fn tiles_in(bytes: usize, cv: usize, o: usize) -> usize {
    bytes / (POINTS * (cv + o) * 4)
}

fn gemm_name(bm: usize, bn: usize) -> &'static str {
    match (bm, bn) {
        (64, 64) => "wino_gemm_64x64",
        (64, 32) => "wino_gemm_64x32",
        _ => "wino_gemm_32x64",
    }
}

/// One Winograd conv: `inputs` are padded activations of `dims` (one, or the decoder's `(up, skip)` pair) whose
/// channels concatenate in order to `u`'s `c`; `out` is `(y, o, bias)`, the padded output of `o` channels whose
/// interior gets the conv plus the bias, when given. The tiles go in chunks of at most `max_tiles` (0: as many as
/// `scratch` holds), each chunk's V and M in `scratch`, whose contents are left undefined; a chunk small enough for
/// the GPU's system-level cache keeps V and M out of DRAM
pub(crate) fn conv(
    gpu: &Gpu,
    u: &Buf,
    inputs: &[(&Buf, usize)],
    (y, o, bias): (&Buf, usize, Option<&Buf>),
    dims: [usize; 3],
    scratch: &Buf,
    max_tiles: usize,
) -> Result<()> {
    ensure!(fits(dims), "Winograd F(4,3) on a level of {dims:?}");
    let cv: usize = inputs.iter().map(|i| i.1).sum();
    ensure!(u.bytes() == POINTS * o * cv * 4, "Winograd weights of {} bytes for {o} x {cv}", u.bytes());
    let [d, h, w] = dims;
    let tiles = (d / 4) * (h / 4) * (w / 4);
    let fit = tiles_in(scratch.bytes(), cv, o);
    ensure!(fit > 0, "Winograd scratch of {} bytes holds no tile of {cv} + {o} channels", scratch.bytes());
    let chunk = if max_tiles == 0 { fit } else { fit.min(max_tiles) };
    let (bias_arg, flags) = match bias {
        Some(b) => (b.arg(), 1u32),
        None => (scratch.arg(), 0),
    };

    let mut t0 = 0;
    while t0 < tiles {
        let nt = chunk.min(tiles - t0);
        let v = scratch.arg();
        let m = scratch.at::<f32>(POINTS * nt * cv);
        let mut coff = 0;
        for &(x, c) in inputs {
            let args =
                [v, x.arg(), d.into(), h.into(), w.into(), c.into(), cv.into(), coff.into(), t0.into(), nt.into()];
            gpu.dispatch("wino_in", [nt, c.div_ceil(32), 1], [192, 1, 1], &args)?;
            coff += c;
        }
        let bn = if o >= 64 { 64 } else { 32 };
        let bm = if bn == 32 || nt >= 64 { 64 } else { 32 };
        gpu.dispatch(
            gemm_name(bm, bn),
            [o.div_ceil(bn), nt.div_ceil(bm), POINTS],
            [128, 1, 1],
            &[v, u.arg(), m, nt.into(), o.into(), cv.into()],
        )?;
        let args =
            [y.arg(), m, bias_arg, d.into(), h.into(), w.into(), o.into(), t0.into(), nt.into(), Arg::U32(flags)];
        gpu.dispatch("wino_out", [nt, o.div_ceil(32), 1], [192, 1, 1], &args)?;
        t0 += nt;
    }

    Ok(())
}

/// Zeroes the padding and slack rows of the padded buffer `x` (`rows` rows of `c` channels around an interior of
/// `dims`) after it served as scratch
pub(crate) fn zero_border(gpu: &Gpu, x: &Buf, dims: [usize; 3], c: usize, rows: usize) -> Result<()> {
    let [d, h, w] = dims;
    let body = (d + 2) * (h + 2) * (w + 2);
    ensure!(rows >= body, "{rows} rows for a padded {dims:?}");
    let count = 2 * (h + 2) * (w + 2) + d * (2 * (w + 2) + 2 * h) + (rows - body);
    gpu.dispatch_1d("zero_border", count * c, &[x.arg(), d.into(), h.into(), w.into(), c.into(), rows.into()])
}
