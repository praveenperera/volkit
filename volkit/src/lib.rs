//! volkit: the shared native GPU runtime for 3D volume networks
//!
//! The CUDA host layer over cudakit ([`gpu`]), channels-last cuDNN convolutions and cuBLAS GEMMs ([`dnn`]), the
//! 3D ResEnc U-Net forward built from an export's layer program ([`unet3d`]), safetensors weights ([`weights`]),
//! zarr IO ([`zarr`], [`unet3d::zio`]) and the nnU-Net contract with AFV v2's seg inference ([`nnunet`]), whose
//! weights can also come straight from an nnU-Net `checkpoint_final.pth` without torch ([`nnunet::export`]). The kernels are cuda-oxide PTX built from `volkit-kernels` and embedded at compile time
//!
//! The code came from inkrs 0.4.0's AFV v2 path with the same numerics: the same cudakit and cudarc, the same cuDNN
//! and cuBLAS calls and algorithm choice, and PTX whose kernels match inkrs's entry for entry. It builds on Linux
//! and Windows; on macOS the crate is empty

#![cfg(not(target_os = "macos"))]

#[cfg(windows)]
pub mod dll;
pub mod dnn;
pub mod error;
pub mod gpu;
pub mod nnunet;
pub mod paths;
pub mod unet3d;
pub mod weights;
pub mod zarr;
