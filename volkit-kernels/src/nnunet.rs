//! The patch input of nnU-Net models run as villa's `vesuvius.predict` runs them, which AFV v2's bf16 mode
//! ([`crate::afv2`]) reuses for its tile input
//!
//! Flip `f` has bit 1 for x, 2 for y, 4 for z: the forward sees voxel `(p-1-z, y, x)` at `(z, y, x)`
//! when bit 4 is set, and so on

use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

/// The coordinate the forward sees at `l` on an axis of `p` when the flip mirrors that axis
#[inline(always)]
fn mirror(l: u32, p: u32, on: bool) -> u32 {
    if on { p - 1 - l } else { l }
}

/// One flip of one patch's model input: `table[code]` (fp16 bits, the normalisation of each uint8 code
/// rounded as villa's dataset casts it) for in-volume voxels, 0 elsewhere (villa zero-pads outside the
/// volume after normalising). Writes channel 0 of `x`, `[p^3, cstride]` fp16 bits; the host zero-fills the
/// other channels once
///
/// The patch's corner is `(oz, oy, ox)` in a uint8 slab of dims `(_, sh, sw)`; patch coordinates in
/// `[v0, v1)` on each axis are inside the volume. `flip` mirrors the patch as above
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn nn_prep(
    mut x: DisjointSlice<u16>,
    slab: &[u8],
    table: &[u16],
    sh: u32,
    sw: u32,
    oz: u32,
    oy: u32,
    ox: u32,
    p: u32,
    vz0: u32,
    vz1: u32,
    vy0: u32,
    vy1: u32,
    vx0: u32,
    vx1: u32,
    cstride: u32,
    flip: u32,
) {
    let e = thread::index_1d_u32(lc).get();
    if e >= p * p * p || table.len() < 256 {
        return;
    }
    let (fx, fy, fz) = (e % p, (e / p) % p, e / (p * p));
    let lx = mirror(fx, p, flip & 1 != 0);
    let ly = mirror(fy, p, flip & 2 != 0);
    let lz = mirror(fz, p, flip & 4 != 0);
    let inside = lz >= vz0 && lz < vz1 && ly >= vy0 && ly < vy1 && lx >= vx0 && lx < vx1;
    let v = if inside { table[slab[(((oz + lz) * sh + oy + ly) * sw + ox + lx) as usize] as usize] } else { 0 };
    let o = (e * cstride) as usize;
    if o < x.len() {
        // SAFETY: in bounds; each thread writes its own voxel's channel 0
        unsafe { *x.get_unchecked_mut(o) = v };
    }
}
