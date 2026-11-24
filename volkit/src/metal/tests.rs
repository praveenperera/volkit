//! The Metal kernels against float64 CPU references, on small shapes that cross every tile edge: partial row and
//! channel tiles, lines longer than a tile, both concat inputs, both stride patterns AFV v2 uses. Each output
//! buffer starts with NaN in every interior element a kernel must write and zero in its padding, so a missed write
//! and a write into the padding both fail. Then the forward on tiny networks, and the shape checks [`Layout`] makes
//! before the device opens. Each test that needs the GPU skips unless `VOLKIT_METAL_TESTS=1`, so a plain
//! `cargo test` never touches it: on a shared Mac the GPU is used only through its job queue

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::gpu::{Arg, Buf, Gpu};
use super::model::{Layout, Model, in_partial, padded_rows};
use super::winograd;
use crate::error::is_unsupported;
use crate::unet3d::host::HostNet;
use crate::unet3d::{Precision, Program};
use crate::weights::Checkpoint;

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

/// A tiny network's program and random weights, built layer by layer
struct Tiny {
    rng: Rng,
    tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Tiny {
    fn tensor(&mut self, name: String, shape: Vec<usize>, scale: f32, offset: f32) -> String {
        let v = self.rng.vec(shape.iter().product()).into_iter().map(|x| offset + scale * x).collect();
        self.tensors.insert(name.clone(), (shape, v));
        name
    }

    /// A `k`^3 conv of stride `s`; the transposed conv's weight is `[cin, cout, k^3]`, a conv's `[cout, cin, k^3]`
    fn conv(&mut self, name: &str, (cin, cout): (usize, usize), k: usize, s: usize, bias: bool, norm: bool) -> Value {
        let shape = if name.ends_with(".t") { vec![cin, cout, k, k, k] } else { vec![cout, cin, k, k, k] };
        let w = self.tensor(format!("{name}.w"), shape, 1.0 / ((cin * k * k * k) as f32).sqrt(), 0.0);
        let b = bias.then(|| self.tensor(format!("{name}.b"), vec![cout], 0.1, 0.0));
        let norm = norm.then(|| {
            let (nw, nb) = (
                self.tensor(format!("{name}.nw"), vec![cout], 0.2, 1.0),
                self.tensor(format!("{name}.nb"), vec![cout], 0.1, 0.0),
            );
            json!({"w": nw, "b": nb, "eps": 1e-5})
        });
        json!({"w": w, "b": b, "cin": cin, "cout": cout, "kernel": [k, k, k], "stride": [s, s, s], "norm": norm, "leaky": null})
    }
}

/// A tiny ResEnc U-Net and its random weights: one block per stage, stage `s` striding by `stages[s].0` into
/// `stages[s].1` channels, a stem of stage 0's width (its bias zero, or absent when `stem_bias` is false), the
/// decoder mirroring the encoder, and a 1x1 head of 4 classes
fn tiny(stages: &[(usize, usize)], stem_bias: bool) -> (Program, Checkpoint) {
    let mut t = Tiny { rng: Rng(11), tensors: BTreeMap::new() };
    let leaky = |mut v: Value| {
        v["leaky"] = json!(0.01);
        v
    };
    let mut c = stages[0].1;
    let mut stem = leaky(t.conv("stem", (1, c), 3, 1, true, true));
    if stem_bias {
        t.tensors.get_mut("stem.b").expect("stem bias").1.fill(0.0);
    } else {
        t.tensors.remove("stem.b");
        stem["b"] = Value::Null;
    }
    let mut encoder = Vec::new();
    for (i, &(s, w)) in stages.iter().enumerate() {
        let conv1 = leaky(t.conv(&format!("e{i}.c1"), (c, w), 3, s, true, true));
        let conv2 = t.conv(&format!("e{i}.c2"), (w, w), 3, 1, true, true);
        let proj = (c != w).then(|| t.conv(&format!("e{i}.p"), (c, w), 1, 1, false, true));
        encoder.push(json!([{"conv1": conv1, "conv2": conv2, "skip": {"pool": s, "proj": proj}, "leaky": 0.01}]));
        c = w;
    }
    let mut decoder = Vec::new();
    for d in 0..stages.len() - 1 {
        let lv = stages.len() - 2 - d;
        let (s, w) = (stages[lv + 1].0, stages[lv].1);
        let transp = t.conv(&format!("d{d}.t"), (c, w), s, s, true, false);
        let conv0 = leaky(t.conv(&format!("d{d}.c0"), (2 * w, w), 3, 1, true, true));
        decoder.push(json!({"transp": transp, "convs": [conv0]}));
        c = w;
    }
    let head = t.conv("head", (c, 4), 1, 1, true, false);
    let prog = json!({"stem": [stem], "stages": encoder, "decoder": decoder, "head": head});
    let prog: Program = serde_json::from_value(prog).expect("program");
    (prog, Checkpoint::from_parts(t.tensors, BTreeMap::new()).expect("checkpoint"))
}

/// AFV v2's program, from the export fixture
fn afv2_program() -> Program {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/nnunet/afv2.json");
    let sidecar: Value = serde_json::from_slice(&std::fs::read(path).expect("fixture")).expect("fixture JSON");
    serde_json::from_value(sidecar["program"].clone()).expect("program")
}

/// AFV v2 keeps Winograd on levels 0 to 2. On a tiny network level 0's scratch (a level-1 buffer of 300 rows x 16
/// channels) holds no tile of its (up, skip) conv, 216 points x (16 + 8) channels, so level 0 runs direct while
/// level 1, whose scratch is level 0's larger buffer, keeps Winograd
#[test]
fn winograd_runs_only_where_its_scratch_holds_a_tile() {
    let afv2 = afv2_program();
    assert_eq!(Layout::new(&afv2, 128, &[0, 1, 2]).expect("afv2").winograd(), [0, 1, 2]);
    let (prog, _) = tiny(&[(1, 8), (2, 16)], true);
    assert_eq!(Layout::new(&prog, 8, &[0, 1, 2]).expect("tiny").winograd(), [1]);
    // with one level there is no level-1 scratch at all
    let (one, _) = tiny(&[(1, 8)], true);
    assert_eq!(Layout::new(&one, 8, &[0]).expect("one level").winograd(), [] as [usize; 0]);
}

/// Shapes the Metal forward can't run fail as unsupported, so a caller can fall back, before the device opens: a
/// width the norm kernels can't take, a strided first stage (whose stem level has no skip for the last decoder
/// stage) and a later stage that keeps its input's resolution
#[test]
fn unsupported_shapes_are_refused_before_the_device_opens() {
    for (stages, p) in [
        (&[(1, 8), (2, 6)][..], 8),
        (&[(2, 8), (1, 8)][..], 8),
        (&[(2, 8), (2, 16)][..], 16),
        (&[(1, 8), (1, 8)][..], 8),
    ] {
        let (prog, _) = tiny(stages, true);
        prog.levels([p; 3], 1).unwrap_or_else(|e| panic!("{stages:?}: the contract accepts it: {e:?}"));
        let err = Layout::new(&prog, p, &[]).err().unwrap_or_else(|| panic!("{stages:?} accepted"));
        assert!(is_unsupported(&err), "{stages:?}: {err:?}");
    }
}

/// The CoreGraphics link `MTLCreateSystemDefaultDevice` needs is in every binary that links volkit: this test
/// binary's load commands name it. Checking the process's loaded images would not do, since Metal loads CoreGraphics
/// on its own; it is the link Apple's docs ask for. `otool` comes with the Command Line Tools, which link this binary
#[test]
fn core_graphics_is_linked() {
    let exe = std::env::current_exe().expect("test binary");
    let out = std::process::Command::new("otool").arg("-L").arg(&exe).output().expect("otool");
    assert!(out.status.success(), "otool -L failed: {}", String::from_utf8_lossy(&out.stderr));
    let libs = String::from_utf8_lossy(&out.stdout);
    assert!(libs.contains("/CoreGraphics.framework/"), "CoreGraphics not linked:\n{libs}");
}

/// The interior of the head's input after one forward of `tile` on [`tiny`]'s network of `stages`, Winograd on
/// `winograd`; each call builds the network afresh, since a checkpoint gives each tensor once
fn forward(
    gpu: &Gpu,
    stages: &[(usize, usize)],
    stem_bias: bool,
    p: usize,
    winograd: &[usize],
    tile: &[u8],
) -> Vec<f32> {
    let (prog, ck) = tiny(stages, stem_bias);
    let layout = Layout::new(&prog, p, winograd).expect("layout");
    assert_eq!(layout.winograd(), winograd, "the layout changed the Winograd levels");
    let net = HostNet::take(&ck, &prog, Precision::Tf32).expect("host weights");
    let mut model = Model::new(gpu, &net, &layout).expect("model");
    let table: Vec<f32> = (0..256).map(|v| (v as f32 - 128.0) / 40.0).collect();
    let (bt, btab) = (gpu.upload(tile).unwrap(), gpu.upload(&table).unwrap());
    model.forward(gpu, bt.arg(), btab.arg()).expect("forward");
    let h = model.head();
    unpad(&host(gpu, h.x, padded_rows([p; 3]) * h.cin), [p; 3], h.cin)
}

/// A tiny network's forward runs with Winograd on the level whose scratch holds a tile and direct convs elsewhere,
/// matching the all-direct forward within Winograd's rounding; a stem without a bias matches one with a zero bias
/// bit for bit
#[test]
fn tiny_forward_runs_and_matches_direct() {
    let Some(gpu) = gpu() else { return };
    let p = 8;
    let mut rng = Rng(12);
    let tile: Vec<u8> = (0..p * p * p).map(|_| ((rng.f() + 1.0) * 127.9) as u8).collect();
    let stages = [(1, 8), (2, 16)];
    let wino = forward(&gpu, &stages, true, p, &[1], &tile);
    let direct = forward(&gpu, &stages, true, p, &[], &tile);
    // f32::max drops NaN differences, so both sides must be finite before the comparison means anything
    assert!(wino.iter().all(|v| v.is_finite()), "Winograd forward is not finite");
    assert!(direct.iter().all(|v| v.is_finite()) && direct.iter().any(|&v| v != 0.0), "degenerate forward");

    let worst = wino.iter().zip(&direct).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("tiny forward: Winograd against direct, worst {worst:.2e}");
    assert!(worst <= 1e-3, "Winograd forward off the direct one by {worst}");

    let no_bias = forward(&gpu, &stages, false, p, &[1], &tile);
    assert!(no_bias.iter().zip(&wino).all(|(a, b)| a.to_bits() == b.to_bits()), "bias-free stem differs");
}
