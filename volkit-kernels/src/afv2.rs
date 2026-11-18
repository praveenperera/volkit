//! Tile input, seg head softmax, blend and column finalisation for AFV v2 as villa's `lasagna/qualzz_inference.py`
//! runs it (`inkrs.AfvV2`), around the float32 forward of [`crate::unet3d::fp32`]
//!
//! Per 128^3 tile the host normalises the uint8 codes with the tile's own statistics into a 256-entry float32
//! table, [`af_prep`] writes the forward's input from it, cuDNN runs the network and its 1x1x1 seg head, and
//! [`af_fold`] adds the head's bias, takes the float32 softmax of the four classes and blends the vertical and
//! horizontal probabilities into the column's float32 accumulators. When every tile of the column is folded,
//! [`af_finish`] normalises the accumulators and writes the `semantic` uint8 output and [`af_pool`] the
//! `tracing` one. Each step is the reference's float32 operation in the reference's order.
//! The opt-in bf16 mode runs the forward on the bf16 glue kernels and takes its input and seg head from
//! [`crate::nnunet::nn_prep`] and [`crate::fiber::nf_head`], then folds and finishes here as the float32 path does.
//! Index math is 32-bit: the host keeps every buffer under 2^32 elements

use core::ptr::addr_of_mut;

use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicU32};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::thread::{blockDim_x, blockIdx_x, sync_threads, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread};

use crate::common::math;

// libdevice's rintf, which `torch.round` compiles to (round half to even)
unsafe extern "C" {
    fn __nv_rintf(x: f32) -> f32;
}

/// Status bit of a column with a non-finite normalised or pooled probability: the reference's `quantize` raises
/// "non-finite probabilities", so the host fails the column before returning any of it
pub const AF_NONFINITE: u32 = 1;

#[inline(always)]
fn finite(v: f32) -> bool {
    let a = if v < 0.0 { -v } else { v };
    a <= 3.402_823_5e38
}

/// Sets `bits` in `stat[idx]`
#[inline(always)]
fn flag(stat: &mut DisjointSlice<u32>, idx: usize, bits: u32) {
    if idx < stat.len() {
        // SAFETY: in bounds; device-scope atomics, only ever setting bits
        unsafe { DeviceAtomicU32::from_ptr(stat.as_mut_ptr().add(idx)).fetch_or(bits, AtomicOrdering::Relaxed) };
    }
}

/// The reference `quantize` of one finite value: `rint(clamp(p, 0, 1) * 255)`, the clamp, multiply and round
/// each a separate float32 operation as `clamp`, `mul_(255)` and `round_()` run them
#[inline(always)]
fn quantize(v: f32) -> u8 {
    let c = math::fminf(math::fmaxf(v, 0.0), 1.0);
    // SAFETY: libdevice's rintf, a pure function
    let r = unsafe { __nv_rintf(mul_rn_f32(c, 255.0)) };
    // `r` is an integer in [0, 255] for finite `v`; a non-finite `v` fails the column on the host
    r as u8
}

/// One tile's model input: `x[e * cstride] = table[tile[e]]`, the tile's normalised codes as float32 into channel
/// 0 of `x` (`[vox, cstride]`); the host zero-fills the other channels once
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn af_prep(mut x: DisjointSlice<f32>, tile: &[u8], table: &[f32], vox: u32, cstride: u32) {
    let e = thread::index_1d_u32(lc).get();
    if e >= vox || (e as usize) >= tile.len() || table.len() < 256 {
        return;
    }
    let o = (e * cstride) as usize;
    if o < x.len() {
        // SAFETY: in bounds; each thread writes its own voxel's channel 0
        unsafe { *x.get_unchecked_mut(o) = table[tile[e as usize] as usize] };
    }
}

/// One tile's seg probabilities blended into the column, as the reference's `out.float().softmax(1)[:, 1:3]`
/// then `acc[:, d] += p[:, s] * w` and `wsum[d] += w` compute them
///
/// - `logits` holds the seg head's four class logits of tile voxel `t` at `t * vstride + k * cstride` float32:
///   the fp32 contract's cuDNN conv output `[p^3, vstride]` (`cstride` 1, classes in its first four channels), or
///   the bf16 mode's `nf_head` output `[4, p^3]` (`vstride` 1, `cstride` `p^3`). `bias` is the head's float32
///   bias, added on its own as PyTorch adds a conv's bias, or empty when the logits already hold it (`nf_head`
///   adds it in bf16, as autocast does);
/// - the softmax takes the classes' maximum, then `exp(l - max)` summed in class order, then each
///   `exp(l - max) / sum`, as for [`crate::fiber::nf_prob`]; classes 1 and 2 (vertical, horizontal) are kept;
/// - `w = (ramp[z] * ramp[y]) * ramp[x]`, the tile weight in the reference's product order;
/// - each product and sum rounded on its own.
///
/// The tile voxels `[s0, s0 + extent)` (per axis, tile coordinates) land on column voxels `[d0, d0 + extent)`;
/// the host clips both to the column, which is `(nz, ny, nx)` with `acc` holding the vertical channel then the
/// horizontal one. One thread per voxel of the extent; tiles are folded one launch at a time in the column's
/// order, so each voxel sums in the reference's order
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn af_fold(
    mut acc: DisjointSlice<f32>,
    mut wsum: DisjointSlice<f32>,
    logits: &[f32],
    bias: &[f32],
    ramp: &[f32],
    p: u32,
    vstride: u32,
    cstride: u32,
    sz: u32,
    sy: u32,
    sx: u32,
    ez: u32,
    ey: u32,
    ex: u32,
    dz: u32,
    dy: u32,
    dx: u32,
    ny: u32,
    nx: u32,
) {
    let e = thread::index_1d_u32(lc).get();
    // the four classes of one voxel must not overlap another voxel's: interleaved (`cstride` 1, `vstride` >= 4) or
    // planar (`vstride` 1, `cstride` >= p^3)
    let planes = vstride >= 4 && cstride == 1 || vstride == 1 && cstride >= p * p * p;
    if e >= ez * ey * ex || ramp.len() < p as usize || !(bias.is_empty() || bias.len() >= 4) || !planes {
        return;
    }
    let (ix, iy, iz) = (e % ex, (e / ex) % ey, e / (ex * ey));
    let (tz, ty, tx) = (sz + iz, sy + iy, sx + ix);
    let t = (tz * p + ty) * p + tx;
    let o = ((dz + iz) * ny + dy + iy) * nx + dx + ix;
    let n = wsum.len() as u32;
    if tz >= p || ty >= p || tx >= p || (t * vstride + 3 * cstride) as usize >= logits.len() || o >= n {
        return;
    }
    if (n + o) as usize >= acc.len() {
        return;
    }
    let (b, c) = ((t * vstride) as usize, cstride as usize);
    let raw = [logits[b], logits[b + c], logits[b + 2 * c], logits[b + 3 * c]];
    let l = if bias.is_empty() {
        raw
    } else {
        [add_rn_f32(raw[0], bias[0]), add_rn_f32(raw[1], bias[1]), add_rn_f32(raw[2], bias[2]), add_rn_f32(raw[3], bias[3])]
    };
    let m = math::fmaxf(math::fmaxf(l[0], l[1]), math::fmaxf(l[2], l[3]));
    let x = [math::expf(l[0] - m), math::expf(l[1] - m), math::expf(l[2] - m), math::expf(l[3] - m)];
    let s = add_rn_f32(add_rn_f32(add_rn_f32(x[0], x[1]), x[2]), x[3]);
    let (pv, ph) = (x[1] / s, x[2] / s);
    let w = mul_rn_f32(mul_rn_f32(ramp[tz as usize], ramp[ty as usize]), ramp[tx as usize]);
    // SAFETY: `o < n <= wsum.len()` and `n + o < acc.len()` by the checks above; one tile's voxels map to distinct
    // column voxels, and one tile's launch finishes before the next queues on the stream
    unsafe {
        let a = acc.get_unchecked_mut(o as usize);
        *a = add_rn_f32(*a, mul_rn_f32(pv, w));
        let a = acc.get_unchecked_mut((n + o) as usize);
        *a = add_rn_f32(*a, mul_rn_f32(ph, w));
        let c = wsum.get_unchecked_mut(o as usize);
        *c = add_rn_f32(*c, w);
    }
}

/// The column's normalisation and `semantic` output, as the reference's `finish` runs them: per voxel `p = acc /
/// wsum` where `wsum > 0` and `acc / 1` elsewhere, in place in `acc` (the tracing pool reads it), then
/// [`quantize`] into `sem`, laid out as `acc` (`[2, n]`). Voxels without weight are counted into `stat[1]` and a
/// non-finite `p` sets [`AF_NONFINITE`] in `stat[0]`. Grid of 256-thread blocks over the `n` voxels; each block
/// tallies its uncovered voxels in shared memory and adds the tally once
#[kernel]
pub fn af_finish(mut sem: DisjointSlice<u8>, mut acc: DisjointSlice<f32>, mut stat: DisjointSlice<u32>, wsum: &[f32]) {
    static mut UNCOVERED: SharedArray<u32, 1> = SharedArray::UNINIT;
    let tid = threadIdx_x();
    let e = blockIdx_x() * blockDim_x() + tid;
    let n = wsum.len() as u32;
    let tally = addr_of_mut!(UNCOVERED) as *mut u32;
    if tid == 0 {
        // SAFETY: thread 0 alone writes the tally before the barrier
        unsafe { *tally = 0 };
    }
    sync_threads();
    // no early return before the barriers: every thread of the block reaches both
    let live = e < n && (2 * n) as usize <= acc.len() && (2 * n) as usize <= sem.len() && stat.len() > 1;
    if live {
        let c = wsum[e as usize];
        let covered = c > 0.0;
        if !covered {
            // SAFETY: the block's shared tally
            unsafe { BlockAtomicU32::from_ptr(tally).fetch_add(1, AtomicOrdering::Relaxed) };
        }
        let d = if covered { c } else { 1.0 };
        let mut bad = false;
        let mut k = 0;
        while k < 2 {
            let o = (k * n + e) as usize;
            // SAFETY: `o < 2n <= acc.len(), sem.len()` by `live`; each thread owns voxel `e` of both channels
            unsafe {
                let a = acc.get_unchecked_mut(o);
                let v = *a / d;
                *a = v;
                bad |= !finite(v);
                *sem.get_unchecked_mut(o) = quantize(v);
            }
            k += 1;
        }
        if bad {
            flag(&mut stat, 0, AF_NONFINITE);
        }
    }
    sync_threads();
    if tid == 0 && stat.len() > 1 {
        // SAFETY: the tally is complete after the barrier
        let count = unsafe { *tally };
        if count != 0 {
            // SAFETY: `1 < stat.len()`; device-scope atomics add the blocks' tallies
            unsafe { DeviceAtomicU32::from_ptr(stat.as_mut_ptr().add(1)).fetch_add(count, AtomicOrdering::Relaxed) };
        }
    }
}

/// The column's `tracing` output, the reference's `quantize(_pool2(p))`: per channel, `(nz / 2, ny / 2, nx / 2)`
/// voxels each the float32 sum, from zero, of its 2x2x2 block of [`af_finish`]'s normalised `p` in `itertools
/// .product((0, 1), repeat=3)` order (z, then y, then x fastest), times 1/8 (exact, as the reference's `/ 8` by
/// a scalar), then [`quantize`]. The odd last plane of an axis is left out, as the reference's slices leave it.
/// A non-finite pooled value sets [`AF_NONFINITE`] in `stat[0]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn af_pool(mut tra: DisjointSlice<u8>, mut stat: DisjointSlice<u32>, p: &[f32], nz: u32, ny: u32, nx: u32) {
    let e = thread::index_1d_u32(lc).get();
    let (hz, hy, hx) = (nz / 2, ny / 2, nx / 2);
    let hn = hz * hy * hx;
    let n = nz * ny * nx;
    if e >= 2 * hn || (2 * hn) as usize > tra.len() || (2 * n) as usize > p.len() {
        return;
    }
    let (k, q) = (e / hn, e % hn);
    let (x, y, z) = (q % hx, (q / hx) % hy, q / (hx * hy));
    let mut s = 0.0f32;
    let mut a = 0;
    while a < 8 {
        let (oz, oy, ox) = (a >> 2, (a >> 1) & 1, a & 1);
        let i = k * n + ((2 * z + oz) * ny + 2 * y + oy) * nx + 2 * x + ox;
        s = add_rn_f32(s, p[i as usize]);
        a += 1;
    }
    let v = mul_rn_f32(s, 0.125);
    if !finite(v) {
        flag(&mut stat, 0, AF_NONFINITE);
    }
    // SAFETY: `e < 2 hn <= tra.len()`; each thread owns one pooled voxel
    unsafe { *tra.get_unchecked_mut(e as usize) = quantize(v) };
}
