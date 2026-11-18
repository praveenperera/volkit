//! cuda-oxide device kernels for volkit
//!
//! `scripts/build-ptx.sh` compiles this crate for sm_80 and sm_120 and writes `volkit/ptx/volkit.sm{80,120}.ptx`,
//! which the volkit crate embeds and loads by kernel name. Buffers are `&[T]` inputs and `DisjointSlice<T>`
//! outputs, each two PTX params (address, then length in elements); scalars are `u32`, `i32` or `f32`. The host
//! checks every launch against the PTX param list.
//!
//! The kernels came from inkrs's kernel crate, unchanged in what they compute: the 3D U-Net glue around cuDNN's
//! channels-last convolutions and cuBLAS's GEMMs ([`unet3d`], bf16 with fp16 and float32 twins), and AFV v2's
//! tile input, softmax, blend and finalisation ([`afv2`]), whose opt-in bf16 mode also takes its input from
//! [`nnunet::nn_prep`] and its seg head from [`fiber::nf_head`]. inkrs's Turing tier is left out: the 3D glue
//! rounds through bf16 for cuDNN's bf16 convs, which need compute capability 8.0

#![no_std]

pub mod afv2;
pub mod common;
pub mod fiber;
pub mod nnunet;
pub mod unet3d;
