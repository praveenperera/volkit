//! volkit: the shared native GPU runtime for 3D volume networks
//!
//! It runs AFV v2's seg inference natively ([`nnunet::afv2`]): the 3D ResEnc U-Net forward over channels-last
//! cuDNN convolutions and cuBLAS GEMMs, built from an export's layer program ([`unet3d`]) and its safetensors
//! weights ([`weights`]), or straight from an nnU-Net model folder without torch ([`nnunet::export`], over the
//! restricted `checkpoint_final.pth` reader [`pth`]). [`dnn`] sets how cuDNN picks its algorithms and caches the
//! choice. The kernels are cuda-oxide PTX built from `volkit-kernels` and embedded at compile time. The device
//! layer under them ([`gpu`], and the cuDNN and cuBLAS calls in [`dnn`]) is public so other runtimes can load
//! their own PTX modules beside volkit's
//!
//! The code came from inkrs 0.4.0's AFV v2 path with the same numerics: the same cudakit and cudarc, the same cuDNN
//! and cuBLAS calls and algorithm choice, and PTX whose kernels match inkrs's entry for entry. It builds on Linux
//! and Windows. On macOS the same engine runs on the Apple GPU ([`nnunet::afv2`] there is a native Metal backend:
//! MSL kernels compiled at load, Metal 4 tensor-op convolutions in float32), sharing the export, checkpoint and
//! weight-layout code

#![warn(missing_docs)]

#[cfg(windows)]
pub mod dll;
#[cfg(not(target_os = "macos"))]
pub mod dnn;
pub mod error;
#[cfg(not(target_os = "macos"))]
pub mod gpu;
#[cfg(target_os = "macos")]
pub(crate) mod metal;
pub mod nnunet;
pub mod paths;
pub mod pth;
pub mod unet3d;
pub mod weights;
