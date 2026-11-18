//! The bf16 head of AFV fiber models' forward, which AFV v2's bf16 mode ([`crate::afv2`]) runs as its seg head
//!
//! Flips follow [`crate::nnunet`]: bit 1 mirrors x, 2 y, 4 z

use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};
use cuda_device::float::{add_rn_f32, fma_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

/// `x` as bf16 holds it, round to nearest even
#[inline(always)]
fn rbf(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16_rne(x))
}

#[inline(always)]
fn mirror(l: u32, p: u32, on: bool) -> u32 {
    if on { p - 1 - l } else { l }
}

/// `nf_head` mode: `sum = logit`
pub const NF_HEAD_SET: u32 = 0;
/// `nf_head` mode: `sum = sum + logit`
pub const NF_HEAD_ADD: u32 = 1;
/// `nf_head` mode: `pair = logit`
pub const NF_HEAD_PAIR_SET: u32 = 2;
/// `nf_head` mode: `sum = sum + (pair + logit)`
pub const NF_HEAD_PAIR_FOLD: u32 = 3;

/// The head of one flip's forward, un-flipped, into the float32 sum over flips
///
/// Per window voxel `e` and class `k`: the forward computed this voxel at its mirror `s`, so `logit =
/// rbf(rbf(sum_c x[s, c] * hw[k, c]) + hb[k])` (PyTorch adds the conv's bias in a separate bf16 op; villa then
/// casts the logits to float32), then `mode` (`NF_HEAD_*`) adds it into `sum[k, e]`, through `pair[k, e]` for
/// the pairwise order of `torch.mean` over the eight flips. `x` is the last decoder stage `[p^3, cols]` bf16
/// pairs; `hw` `[classes, cols]` and `hb` `[classes]` are the head's bf16 weight and bias as float32
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn nf_head(
    mut sum: DisjointSlice<f32>,
    mut pair: DisjointSlice<f32>,
    x: &[u32],
    hw: &[f32],
    hb: &[f32],
    cols: u32,
    classes: u32,
    p: u32,
    flip: u32,
    mode: u32,
) {
    let e = thread::index_1d_u32(lc).get();
    let pairs = cols / 2;
    let vox = p * p * p;
    if e >= vox || pairs == 0 || hw.len() < (classes * cols) as usize || hb.len() < classes as usize {
        return;
    }
    let (lx, ly, lz) = (e % p, (e / p) % p, e / (p * p));
    let sx = mirror(lx, p, flip & 1 != 0);
    let sy = mirror(ly, p, flip & 2 != 0);
    let sz = mirror(lz, p, flip & 4 != 0);
    let s = (sz * p + sy) * p + sx;
    let mut k = 0;
    while k < classes {
        let mut dot = 0f32;
        let mut cp = 0;
        while cp < pairs {
            let v = x[(s * pairs + cp) as usize];
            dot = fma_rn_f32(bf16_to_f32(v as u16), hw[(k * cols + 2 * cp) as usize], dot);
            dot = fma_rn_f32(bf16_to_f32((v >> 16) as u16), hw[(k * cols + 2 * cp + 1) as usize], dot);
            cp += 1;
        }
        let logit = rbf(add_rn_f32(rbf(dot), hb[k as usize]));
        let o = (k * vox + e) as usize;
        if o < sum.len() && o < pair.len() {
            // SAFETY: in bounds; each thread owns voxel `e` of every class
            unsafe {
                let a = sum.get_unchecked_mut(o);
                let b = pair.get_unchecked_mut(o);
                match mode {
                    NF_HEAD_SET => *a = logit,
                    NF_HEAD_ADD => *a = add_rn_f32(*a, logit),
                    NF_HEAD_PAIR_SET => *b = logit,
                    _ => *a = add_rn_f32(*a, add_rn_f32(*b, logit)),
                }
            }
        }
        k += 1;
    }
}
