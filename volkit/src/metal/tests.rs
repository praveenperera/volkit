//! The Metal kernels against float64 CPU references, on small shapes that cross every tile edge: partial row and
//! channel tiles, lines longer than a tile, both concat inputs, both stride patterns AFV v2 uses. Each output
//! buffer starts with NaN in every interior element a kernel must write and zero in its padding, so a missed write
//! and a write into the padding both fail. Each test needs the GPU and skips unless `VOLKIT_METAL_TESTS=1`, so a
//! plain `cargo test` never touches it: on a shared Mac the GPU is used only through its job queue

use super::gpu::{Arg, Buf, Gpu};
use super::model::{in_partial, padded_rows};
use super::winograd;

/// The GPU when `VOLKIT_METAL_TESTS=1`, else `None` and the test returns early
fn gpu() -> Option<Gpu> {
    if std::env::var_os("VOLKIT_METAL_TESTS").is_none_or(|v| v != "1") {
        eprintln!("skipped: set VOLKIT_METAL_TESTS=1 to run the Metal kernel tests on the GPU");
        return None;
    }
    Some(Gpu::open(0).expect("gpu"))
}

/// A deterministic uniform stream in [-1, 1)
struct Rng(u64);

impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f()).collect()
    }
}

fn prow(z: usize, y: usize, x: usize, h: usize, w: usize) -> usize {
    ((z + 1) * (h + 2) + y + 1) * (w + 2) + x + 1
}

/// Plain `[d, h, w, c]` into the padded layout with its slack rows; `fill` goes in every interior element
fn pad(x: Option<&[f32]>, d: [usize; 3], c: usize, fill: f32) -> Vec<f32> {
    let mut out = vec![0f32; padded_rows(d) * c];
    for z in 0..d[0] {
        for y in 0..d[1] {
            for xx in 0..d[2] {
                let o = prow(z, y, xx, d[1], d[2]) * c;
                let i = ((z * d[1] + y) * d[2] + xx) * c;
                match x {
                    Some(x) => out[o..o + c].copy_from_slice(&x[i..i + c]),
                    None => out[o..o + c].fill(fill),
                }
            }
        }
    }
    out
}

/// The interior of a padded buffer as plain `[d, h, w, c]`, after checking every other row is still zero
fn unpad(buf: &[f32], d: [usize; 3], c: usize) -> Vec<f32> {
    let mut interior = vec![false; padded_rows(d)];
    let mut out = Vec::with_capacity(d.iter().product::<usize>() * c);
    for z in 0..d[0] {
        for y in 0..d[1] {
            for x in 0..d[2] {
                let r = prow(z, y, x, d[1], d[2]);
                interior[r] = true;
                out.extend_from_slice(&buf[r * c..(r + 1) * c]);
            }
        }
    }
    for (r, inside) in interior.iter().enumerate() {
        if !inside {
            assert!(buf[r * c..(r + 1) * c].iter().all(|&v| v == 0.0), "padding row {r} was written");
        }
    }
    out
}

/// Every element within `tol` of the float64 reference, relative to the sum of absolute terms behind it
fn check(name: &str, got: &[f32], want: &[f64], scale: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut worst = (0.0f64, 0usize);
    for (i, ((&g, &w), &s)) in got.iter().zip(want).zip(scale).enumerate() {
        assert!(g.is_finite(), "{name}: element {i} is {g} (never written?)");
        let err = (f64::from(g) - w).abs() / (s + 1e-30);
        if err > worst.0 {
            worst = (err, i);
        }
    }
    eprintln!("{name}: worst {:.2e} of its scale", worst.0);
    assert!(worst.0 <= tol, "{name}: element {} off by {:.2e} of its scale (tol {tol:.0e})", worst.1, worst.0);
}

fn host(gpu: &Gpu, buf: &Buf, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    gpu.download(buf, 0, &mut out).expect("download");
    out
}

/// A 3x3x3 pad-1 conv in float64: x plain `[d, c]`, weights `[o, 27, c]`; output dims ceil(d / s); returns the
/// output and each output's sum of absolute terms
fn conv_ref(
    x: &[f32],
    d: [usize; 3],
    c: usize,
    w: &[f32],
    o: usize,
    bias: Option<&[f32]>,
    s: [usize; 3],
) -> (Vec<f64>, Vec<f64>, [usize; 3]) {
    let od: [usize; 3] = std::array::from_fn(|a| d[a].div_ceil(s[a]));
    let mut out = Vec::new();
    let mut scale = Vec::new();
    for z in 0..od[0] {
        for y in 0..od[1] {
            for xx in 0..od[2] {
                for k in 0..o {
                    let (mut acc, mut abs) = (0f64, 0f64);
                    for t in 0..27 {
                        let iz = (s[0] * z + t / 9) as isize - 1;
                        let iy = (s[1] * y + (t / 3) % 3) as isize - 1;
                        let ix = (s[2] * xx + t % 3) as isize - 1;
                        if iz < 0
                            || iy < 0
                            || ix < 0
                            || iz >= d[0] as isize
                            || iy >= d[1] as isize
                            || ix >= d[2] as isize
                        {
                            continue;
                        }
                        let base = ((iz as usize * d[1] + iy as usize) * d[2] + ix as usize) * c;
                        for ci in 0..c {
                            let p = f64::from(x[base + ci]) * f64::from(w[(k * 27 + t) * c + ci]);
                            acc += p;
                            abs += p.abs();
                        }
                    }
                    if let Some(b) = bias {
                        acc += f64::from(b[k]);
                        abs += f64::from(b[k]).abs();
                    }
                    out.push(acc);
                    scale.push(abs);
                }
            }
        }
    }
    (out, scale, od)
}

fn add_into(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// The kernel library compiles and every kernel passes the argument check; no kernel runs, so this one may run
/// outside the GPU queue
#[test]
fn library_compiles() {
    let Some(gpu) = gpu() else { return };
    gpu.ready().expect("kernel library");
}

/// taps_s1 in every tile shape, with and without the concat input and the bias: lines of 70 voxels (two row tiles
/// of 64 a line), 48 output channels (a partial 32-wide channel tile)
#[test]
fn taps_s1_matches_float64() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(1);
    let (d, c, o) = ([4usize, 5, 70], 32usize, 48usize);
    let vox: usize = d.iter().product();
    let x = rng.vec(vox * c);
    let x2 = rng.vec(vox * c);
    let w = rng.vec(o * 27 * c);
    let w2 = rng.vec(o * 27 * c);
    let bias = rng.vec(o);
    let (mut want, mut scale, _) = conv_ref(&x, d, c, &w, o, Some(&bias), [1; 3]);
    let (want2, scale2, _) = conv_ref(&x2, d, c, &w2, o, None, [1; 3]);
    let (plain, plain_scale, _) = conv_ref(&x, d, c, &w, o, None, [1; 3]);
    add_into(&mut want, &want2);
    add_into(&mut scale, &scale2);
    let rows = padded_rows(d);
    let bx = gpu.upload(&pad(Some(&x), d, c, 0.0)).unwrap();
    let bx2 = gpu.upload(&pad(Some(&x2), d, c, 0.0)).unwrap();
    let (bw, bw2, bb) = (gpu.upload(&w).unwrap(), gpu.upload(&w2).unwrap(), gpu.upload(&bias).unwrap());
    let span = d[1] * (d[2] + 2);
    for (name, bm, bn) in
        [("taps_s1_64x64", 64, 64), ("taps_s1_64x32", 64, 32), ("taps_s1_32x64", 32, 64), ("taps_s1_32x32", 32, 32)]
    {
        for concat in [false, true] {
            let by = gpu.upload(&pad(None, d, o, f32::NAN)).unwrap();
            let (xa, wa, c2, flags) = if concat { (&bx2, &bw2, c, 1u32) } else { (&bx, &bw, 0, 0) };
            let args: [Arg; 14] = [
                bx.arg(),
                bw.arg(),
                xa.arg(),
                wa.arg(),
                by.arg(),
                bb.arg(),
                d[0].into(),
                d[1].into(),
                d[2].into(),
                c.into(),
                c2.into(),
                o.into(),
                rows.into(),
                Arg::U32(flags),
            ];
            gpu.dispatch(name, [o.div_ceil(bn), d[0] * span.div_ceil(bm), 1], [128, 1, 1], &args).unwrap();
            let got = unpad(&host(&gpu, &by, rows * o), d, o);
            let label = format!("{name} concat={concat}");
            if concat {
                check(&label, &got, &want, &scale, 1e-6);
            } else {
                check(&label, &got, &plain, &plain_scale, 1e-6);
            }
        }
    }
}

/// taps_sn at AFV v2's strides, (2, 2, 2) and the last stage's (2, 2, 1), with the bias
#[test]
fn taps_sn_matches_float64() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(2);
    let (d, c, o) = ([6usize, 8, 70], 32usize, 64usize);
    let x = rng.vec(d.iter().product::<usize>() * c);
    let w = rng.vec(o * 27 * c);
    let bias = rng.vec(o);
    let bx = gpu.upload(&pad(Some(&x), d, c, 0.0)).unwrap();
    let (bw, bb) = (gpu.upload(&w).unwrap(), gpu.upload(&bias).unwrap());
    for s in [[2usize, 2, 2], [2, 2, 1]] {
        let (want, scale, od) = conv_ref(&x, d, c, &w, o, Some(&bias), s);
        for (name, bm, bn) in [("taps_sn_64x64", 64, 64), ("taps_sn_32x32", 32, 32)] {
            let by = gpu.upload(&pad(None, od, o, f32::NAN)).unwrap();
            let args: [Arg; 16] = [
                bx.arg(),
                bw.arg(),
                by.arg(),
                bb.arg(),
                d[0].into(),
                d[1].into(),
                d[2].into(),
                c.into(),
                od[0].into(),
                od[1].into(),
                od[2].into(),
                o.into(),
                s[0].into(),
                s[1].into(),
                s[2].into(),
                Arg::U32(1),
            ];
            gpu.dispatch(name, [o.div_ceil(bn), od[0] * od[1] * od[2].div_ceil(bm), 1], [128, 1, 1], &args).unwrap();
            let got = unpad(&host(&gpu, &by, padded_rows(od) * o), od, o);
            check(&format!("{name} stride {s:?}"), &got, &want, &scale, 1e-6);
        }
    }
}

/// gemm_nt over a partial last row tile and a partial channel tile
#[test]
fn gemm_nt_matches_float64() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(3);
    let (m, n, k) = (1000usize, 48usize, 32usize);
    let x = rng.vec(m * k);
    let w = rng.vec(n * k);
    let (mut want, mut scale) = (Vec::new(), Vec::new());
    for r in 0..m {
        for col in 0..n {
            let terms = (0..k).map(|i| f64::from(x[r * k + i]) * f64::from(w[col * k + i]));
            want.push(terms.clone().sum());
            scale.push(terms.map(f64::abs).sum());
        }
    }
    let (bx, bw) = (gpu.upload(&x).unwrap(), gpu.upload(&w).unwrap());
    for (name, bn) in [("gemm_nt_64x64", 64), ("gemm_nt_64x32", 32)] {
        let by = gpu.upload(&vec![f32::NAN; m * n]).unwrap();
        let args = [bx.arg(), bw.arg(), by.arg(), m.into(), n.into(), k.into()];
        gpu.dispatch(name, [n.div_ceil(bn), m.div_ceil(64), 1], [128, 1, 1], &args).unwrap();
        check(name, &host(&gpu, &by, m * n), &want, &scale, 1e-6);
    }
}

/// tconv at both upsampling factors: every fine voxel is written once, with its tap's weights and the bias
#[test]
fn tconv_matches_float64() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(4);
    let (d, k, cout) = ([3usize, 4, 5], 64usize, 32usize);
    let x = rng.vec(d.iter().product::<usize>() * k);
    let bias = rng.vec(cout);
    let bx = gpu.upload(&pad(Some(&x), d, k, 0.0)).unwrap();
    let bb = gpu.upload(&bias).unwrap();
    let in_rows = (d[0] + 2) * (d[1] + 2) * (d[2] + 2);
    for f in [[2usize, 2, 2], [2, 2, 1]] {
        let taps: usize = f.iter().product();
        let n = taps * cout;
        let w = rng.vec(n * k);
        let od = [f[0] * d[0], f[1] * d[1], f[2] * d[2]];
        let (mut want, mut scale) = (Vec::new(), Vec::new());
        for z in 0..od[0] {
            for y in 0..od[1] {
                for xx in 0..od[2] {
                    let tap = ((z % f[0]) * f[1] + y % f[1]) * f[2] + xx % f[2];
                    let src = (((z / f[0]) * d[1] + y / f[1]) * d[2] + xx / f[2]) * k;
                    for co in 0..cout {
                        let terms = (0..k).map(|i| f64::from(x[src + i]) * f64::from(w[(tap * cout + co) * k + i]));
                        want.push(terms.clone().sum::<f64>() + f64::from(bias[co]));
                        scale.push(terms.map(f64::abs).sum::<f64>() + f64::from(bias[co]).abs());
                    }
                }
            }
        }
        let bw = gpu.upload(&w).unwrap();
        for (name, bn) in [("tconv_64x64", 64), ("tconv_64x32", 32)] {
            let by = gpu.upload(&pad(None, od, cout, f32::NAN)).unwrap();
            let args: [Arg; 12] = [
                bx.arg(),
                bw.arg(),
                by.arg(),
                bb.arg(),
                d[0].into(),
                d[1].into(),
                d[2].into(),
                k.into(),
                cout.into(),
                f[0].into(),
                f[1].into(),
                f[2].into(),
            ];
            gpu.dispatch(name, [n.div_ceil(bn), in_rows.div_ceil(64), 1], [128, 1, 1], &args).unwrap();
            let got = unpad(&host(&gpu, &by, padded_rows(od) * cout), od, cout);
            check(&format!("{name} factor {f:?}"), &got, &want, &scale, 1e-6);
        }
    }
}

/// Per-channel mean and biased variance in float64
fn moments(x: &[f32], c: usize) -> Vec<(f64, f64)> {
    let rows = x.len() / c;
    (0..c)
        .map(|ch| {
            let mean = (0..rows).map(|r| f64::from(x[r * c + ch])).sum::<f64>() / rows as f64;
            let var = (0..rows).map(|r| (f64::from(x[r * c + ch]) - mean).powi(2)).sum::<f64>() / rows as f64;
            (mean, var)
        })
        .collect()
}

/// InstanceNorm statistics, then the apply with each residual kind and LeakyReLU, on 40 channels (10 quads, a group
/// that is not a power of two) over 4224 lines, more than `MAX_CHUNKS`, so a chunk takes two lines and its lanes sum
/// different counts
#[test]
fn instance_norm_matches_float64() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(5);
    let (d, c) = ([66usize, 64, 30], 40usize);
    let vox: usize = d.iter().product();
    // an offset mean and a spread, as conv outputs have
    let y: Vec<f32> =
        rng.vec(vox * c).iter().enumerate().map(|(i, v)| 3.0 + v * (1.0 + (i % c) as f32 * 0.1)).collect();
    let res = rng.vec(vox * c);
    let rs: Vec<f32> = rng.vec(vox * c).iter().map(|v| v * 4.0 - 1.0).collect();
    let (gamma, beta, rgamma, rbeta) = (rng.vec(c), rng.vec(c), rng.vec(c), rng.vec(c));
    let (eps, slope) = (1e-5f32, 0.01f32);
    let (m, rm) = (moments(&y, c), moments(&rs, c));
    let rows = padded_rows(d);

    let part = gpu.alloc(4096 * c * 16).unwrap();
    let stats_of = |x: &Buf, padded: bool, out: &Buf| {
        let chunks = in_partial(&gpu, &part, x, d, c, padded).unwrap();
        let args = [out.arg(), part.arg(), chunks.into(), c.into(), eps.into()];
        gpu.dispatch("in_finalize", [c, 1, 1], [256, 1, 1], &args).unwrap();
    };
    let (bg, bbeta, brg, brb) = (
        gpu.upload(&gamma).unwrap(),
        gpu.upload(&beta).unwrap(),
        gpu.upload(&rgamma).unwrap(),
        gpu.upload(&rbeta).unwrap(),
    );
    let bres = gpu.upload(&pad(Some(&res), d, c, 0.0)).unwrap();
    let brs = gpu.upload(&rs).unwrap();
    let rstats = gpu.alloc(c * 8).unwrap();
    stats_of(&brs, false, &rstats);

    // modes: none, padded plain residual, normalised projection, plain pooled residual
    let bres_plain = gpu.upload(&res).unwrap();
    for mode in 0u32..4 {
        let by = gpu.upload(&pad(Some(&y), d, c, 0.0)).unwrap();
        let stats = gpu.alloc(c * 8).unwrap();
        stats_of(&by, true, &stats);
        let mut s = vec![0f32; 2 * c];
        gpu.download(&stats, 0, &mut s).unwrap();
        for ch in 0..c {
            let (mean, var) = m[ch];
            assert!(
                (f64::from(s[2 * ch]) - mean).abs() <= 1e-5 * mean.abs(),
                "mean of channel {ch}: {} vs {mean}",
                s[2 * ch]
            );
            let inv = 1.0 / (var + f64::from(eps)).sqrt();
            assert!((f64::from(s[2 * ch + 1]) - inv).abs() <= 1e-5 * inv, "invstd of channel {ch}");
        }
        let resb = match mode {
            1 => &bres,
            3 => &bres_plain,
            _ => &brs,
        };
        let args: [Arg; 15] = [
            by.arg(),
            stats.arg(),
            bg.arg(),
            bbeta.arg(),
            resb.arg(),
            rstats.arg(),
            brg.arg(),
            brb.arg(),
            d[0].into(),
            d[1].into(),
            d[2].into(),
            c.into(),
            Arg::U32(mode),
            Arg::U32(1),
            slope.into(),
        ];
        gpu.dispatch_1d("in_apply", vox * c / 4, &args).unwrap();
        let got = unpad(&host(&gpu, &by, rows * c), d, c);
        let (mut want, mut scale) = (Vec::new(), Vec::new());
        for r in 0..vox {
            for ch in 0..c {
                let i = r * c + ch;
                let (mean, var) = m[ch];
                let mut v = f64::from(gamma[ch]) * (f64::from(y[i]) - mean) / (var + f64::from(eps)).sqrt()
                    + f64::from(beta[ch]);
                let mut sc = v.abs() + f64::from(beta[ch]).abs() + 1.0;
                match mode {
                    1 | 3 => {
                        v += f64::from(res[i]);
                        sc += f64::from(res[i]).abs();
                    }
                    2 => {
                        let (rmean, rvar) = rm[ch];
                        let t = f64::from(rgamma[ch]) * (f64::from(rs[i]) - rmean) / (rvar + f64::from(eps)).sqrt()
                            + f64::from(rbeta[ch]);
                        v += t;
                        sc += t.abs();
                    }
                    _ => {}
                }
                want.push(if v > 0.0 { v } else { v * f64::from(slope) });
                scale.push(sc);
            }
        }
        check(&format!("in_apply mode {mode}"), &got, &want, &scale, 1e-5);
    }
}

/// The stem from uint8 codes through the table, at the tile's edges too
#[test]
fn stem_matches_float64() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(6);
    let (p, o) = (10usize, 32usize);
    let tile: Vec<u8> = (0..p.pow(3)).map(|_| ((rng.f() + 1.0) * 127.9) as u8).collect();
    let table = rng.vec(256);
    let w = rng.vec(o * 27);
    let bias = rng.vec(o);
    let x: Vec<f32> = tile.iter().map(|&v| table[v as usize]).collect();
    let (want, scale, _) = conv_ref(&x, [p; 3], 1, &w, o, Some(&bias), [1; 3]);
    let by = gpu.upload(&pad(None, [p; 3], o, f32::NAN)).unwrap();
    let (bt, btab, bw, bb) =
        (gpu.upload(&tile).unwrap(), gpu.upload(&table).unwrap(), gpu.upload(&w).unwrap(), gpu.upload(&bias).unwrap());
    let count = p.pow(3) * o / 8;
    let args = [by.arg(), bt.arg(), btab.arg(), bw.arg(), bb.arg(), p.into(), o.into(), count.into()];
    gpu.dispatch_1d("stem", count, &args).unwrap();
    let got = unpad(&host(&gpu, &by, padded_rows([p; 3]) * o), [p; 3], o);
    check("stem", &got, &want, &scale, 1e-6);
}

/// avgpool's mean of each block, from a padded input into a plain output
#[test]
fn avgpool_matches_float64() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(7);
    let (od, c, f) = ([3usize, 4, 5], 32usize, [2usize, 2, 1]);
    let d = [od[0] * f[0], od[1] * f[1], od[2] * f[2]];
    let x = rng.vec(d.iter().product::<usize>() * c);
    let (mut want, mut scale) = (Vec::new(), Vec::new());
    for z in 0..od[0] {
        for y in 0..od[1] {
            for xx in 0..od[2] {
                for ch in 0..c {
                    let mut s = 0f64;
                    let mut a = 0f64;
                    for dz in 0..f[0] {
                        for dy in 0..f[1] {
                            for dx in 0..f[2] {
                                let v = f64::from(
                                    x[(((f[0] * z + dz) * d[1] + f[1] * y + dy) * d[2] + f[2] * xx + dx) * c + ch],
                                );
                                s += v;
                                a += v.abs();
                            }
                        }
                    }
                    want.push(s / 4.0);
                    scale.push(a);
                }
            }
        }
    }
    let bx = gpu.upload(&pad(Some(&x), d, c, 0.0)).unwrap();
    let n = od.iter().product::<usize>() * c;
    let by = gpu.upload(&vec![f32::NAN; n]).unwrap();
    let args =
        [by.arg(), bx.arg(), od[0].into(), od[1].into(), od[2].into(), c.into(), f[0].into(), f[1].into(), f[2].into()];
    gpu.dispatch_1d("avgpool", n, &args).unwrap();
    check("avgpool", &host(&gpu, &by, n), &want, &scale, 1e-6);
}

/// Winograd F(4,3) against the float64 conv: one input and the decoder's concat pair, 48 output channels (a partial
/// channel group and GEMM tile), through a scratch too small for the level, so it runs in two chunks. F(4,3)'s
/// transforms scale values by up to about 100 and round in float32, hence the looser bound than the tap convs'
#[test]
fn winograd_matches_float64() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(8);
    let (d, c, o) = ([8usize, 8, 12], 32usize, 48usize);
    let vox: usize = d.iter().product();
    let (x, x2) = (rng.vec(vox * c), rng.vec(vox * c));
    let bias = rng.vec(o);
    let bx = gpu.upload(&pad(Some(&x), d, c, 0.0)).unwrap();
    let bx2 = gpu.upload(&pad(Some(&x2), d, c, 0.0)).unwrap();
    let bb = gpu.upload(&bias).unwrap();
    let per_plane = (d[1] / 4) * (d[2] / 4);
    for concat in [false, true] {
        let cv = if concat { 2 * c } else { c };
        // the full weight over the concatenated input, [o, 27, cv]
        let w = rng.vec(o * 27 * cv);
        let xs: Vec<f32> = if concat {
            (0..vox).flat_map(|v| x[v * c..(v + 1) * c].iter().chain(&x2[v * c..(v + 1) * c]).copied()).collect()
        } else {
            x.clone()
        };
        let (want, scale, _) = conv_ref(&xs, d, cv, &w, o, Some(&bias), [1; 3]);
        let u = gpu.upload(&winograd::transform_weights(&w, o, cv)).unwrap();
        // room for one and a half planes of tiles
        let scratch = gpu.alloc(winograd::POINTS * (cv + o) * 4 * per_plane * 3 / 2).unwrap();
        let inputs: Vec<(&Buf, usize)> = if concat { vec![(&bx, c), (&bx2, c)] } else { vec![(&bx, c)] };
        // chunks as large as the scratch allows (9 tiles, then 3) and of 5 tiles, both splitting planes of 6
        for max_tiles in [0, 5] {
            let by = gpu.upload(&pad(None, d, o, f32::NAN)).unwrap();
            winograd::conv(&gpu, &u, &inputs, (&by, o, Some(&bb)), d, &scratch, max_tiles).unwrap();
            let got = unpad(&host(&gpu, &by, padded_rows(d) * o), d, o);
            check(&format!("winograd concat={concat} max_tiles={max_tiles}"), &got, &want, &scale, 2e-5);
        }
    }
}

/// zero_border clears every padding and slack row and leaves the interior alone
#[test]
fn zero_border_clears_only_the_padding() {
    let Some(gpu) = gpu() else { return };
    let (d, c) = ([4usize, 5, 6], 8usize);
    let rows = padded_rows(d);
    let buf = gpu.upload(&vec![1.0f32; rows * c]).unwrap();
    winograd::zero_border(&gpu, &buf, d, c, rows).unwrap();
    let got = host(&gpu, &buf, rows * c);
    // unpad asserts the padding is zero; the interior must be untouched
    assert!(unpad(&got, d, c).iter().all(|&v| v == 1.0), "zero_border wrote an interior voxel");
}

/// The validation run's negative control: zero_border on a buffer half the size its shape needs stores far past the
/// buffer's end, which shader validation must report. Runs only with `VOLKIT_METAL_OOB_CONTROL=1`, since without the
/// validation layer the stores land in whatever memory follows the buffer
#[test]
fn oob_control_stores_past_the_buffer() {
    if std::env::var_os("VOLKIT_METAL_OOB_CONTROL").is_none_or(|v| v != "1") {
        return;
    }

    let Some(gpu) = gpu() else { return };
    let (d, c) = ([16usize, 16, 16], 32usize);
    let rows = padded_rows(d);
    // about 380 KB short, well past any page rounding of the allocation
    let buf = gpu.upload(&vec![1.0f32; rows * c / 2]).unwrap();
    winograd::zero_border(&gpu, &buf, d, c, rows).unwrap();
    let _ = gpu.synchronize();
}
