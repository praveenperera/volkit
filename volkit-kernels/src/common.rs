//! Helpers shared by the kernels: libdevice math and the FP16 conversions
//!
//! Taken from inkrs's `common.rs` without its Turing (sm_75) fallbacks and the 2D readers' FP16 packing and
//! tensor-core helpers: every kernel here needs compute capability 8.0, so volkit builds the sm_80 and sm_120
//! tiers only, from the code inkrs compiles for them

use cuda_device::convert::{cvt_f16x2_f32, cvt_f32_f16x2_lo};

// the CUDA C++ math library calls these same libdevice functions, so results match what
// PyTorch's nvcc-built kernels compute with `std::exp` and `::sqrtf`
unsafe extern "C" {
    fn __nv_expf(x: f32) -> f32;
    fn __nv_sqrtf(x: f32) -> f32;
    fn __nv_fmaxf(x: f32, y: f32) -> f32;
    fn __nv_fminf(x: f32, y: f32) -> f32;
}

/// Math functions under their CUDA C++ names
pub mod math {
    #[inline(always)]
    pub fn expf(x: f32) -> f32 {
        unsafe { super::__nv_expf(x) }
    }

    #[inline(always)]
    pub fn sqrtf(x: f32) -> f32 {
        unsafe { super::__nv_sqrtf(x) }
    }

    /// One `max.f32`; `f32::max` lowers to a compare and a select on some routes
    #[inline(always)]
    pub fn fmaxf(x: f32, y: f32) -> f32 {
        unsafe { super::__nv_fmaxf(x, y) }
    }

    /// One `min.f32`, as [`fmaxf`]
    #[inline(always)]
    pub fn fminf(x: f32, y: f32) -> f32 {
        unsafe { super::__nv_fminf(x, y) }
    }
}

/// One FP16 value, as raw bits, widened to float
#[inline(always)]
pub fn widen(bits: u16) -> f32 {
    cvt_f32_f16x2_lo(bits as u32)
}

/// One float rounded to nearest-even FP16, as raw bits
#[inline(always)]
pub fn narrow(x: f32) -> u16 {
    cvt_f16x2_f32(x, 0.0) as u16
}
