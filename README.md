# volkit

Shared native GPU runtime for 3D volume networks: the CUDA host layer over cudakit, cuDNN and cuBLAS layers, PTX kernels, blending, a Metal backend for Apple silicon, and the networks inkrs and fibrs run on it.

It currently holds what native AFV fibre model V2 inference needs: the AFV v2 engine (`volkit::nnunet::afv2::AfvV2`), the 3D ResEnc U-Net forward it runs, and a reader for nnU-Net `checkpoint_final.pth` files that needs no torch or Python. The code and kernels came from [inkrs](https://github.com/praveenperera/inkrs) 0.4.0's AFV v2 path and compute the same output.

## Use

```toml
[dependencies]
volkit = "0.2"
```

```rust
use std::path::Path;

use volkit::nnunet::afv2::{AfvPrecision, AfvV2, Options};

/// One column of `shape` (nz, ny, nx) at band position (y0, x0): `tiles` holds up to `MAX_BATCH` 128^3 uint8 tiles
/// one after another, `origins` their padded lattice origins
fn column(
    model_dir_or_export: &Path,
    shape: [usize; 3],
    (y0, x0): (usize, usize),
    tiles: &[u8],
    origins: &[[usize; 3]],
) -> color_eyre::Result<()> {
    // CUDA's options also choose how cuDNN picks its algorithms; macOS's Metal engine has no cuDNN
    #[cfg(not(target_os = "macos"))]
    let opts = Options { device: 0, pick: volkit::dnn::Pick::Find, precision: AfvPrecision::Fp32 };
    #[cfg(target_os = "macos")]
    let opts = Options { device: 0, precision: AfvPrecision::Fp32 };

    let mut net = AfvV2::open(model_dir_or_export, &opts)?;
    net.begin(shape, y0, x0)?;
    net.add(tiles, origins)?;
    // semantic (2, nz, ny, nx) and tracing (2, nz/2, ny/2, nx/2) uint8
    let column = net.finish(true)?;
    Ok(())
}
```

[docs/API.md](https://github.com/praveenperera/volkit/blob/master/docs/API.md) lists the public API and environment variables.

## Precision

volkit defaults to parity with the current best implementation, the public reference runner (villa's `lasagna.qualzz_inference` on PyTorch).

- **`AfvPrecision::Fp32`, the default**, runs at the reference's precision: float32 tensors with TF32 tensor-core convolutions and GEMMs, as PyTorch's float32 defaults run the model. Its output is byte-identical to inkrs 0.4.0's AFV v2 engine.
- **On Metal, `Fp32`** runs the convolutions and GEMMs with true float32 operands, where CUDA uses TF32, so its output sits inside TF32's rounding noise of CUDA's rather than byte-identical to it.
- **`AfvPrecision::Bf16`** (CUDA only) is an opt-in speed mode with small, measured differences. It is never the default and is not used for published fibre sets. Measured against float32 on two CT bands with vertical and horizontal fibres (driver 595.112.07), tracing fibres on both outputs:
  - about 99% of traced fibre length agrees (99.2-99.4%);
  - 97.0-97.8% of fibres match;
  - fibre counts are within 0.3%;
  - over the foreground, the 99.9th percentile of the absolute difference in the output fields is 7 to 11 of 255.

Every speed figure for volkit names the precision it was measured at.

## Requirements

- **CUDA**: Linux or Windows x86_64 with an NVIDIA GPU of compute capability 8.0 or above (RTX 30 / A100 and later). At run time: the NVIDIA driver, cuDNN 9 and cuBLAS, loaded with `dlopen` and checked before first use. Building needs no CUDA toolkit.
- **Metal**: macOS 26 or newer (Metal 4) on Apple silicon (M1 or newer). The kernels are MSL compiled when the engine opens, so building needs neither Xcode nor the Metal toolchain. Tested on an M4 Pro with macOS 27.2.

macOS builds the Metal backend in place of the CUDA one; the two expose the same `AfvV2` API, apart from `Options` and `LoadReport`'s fields.

### Library versions on Linux

volkit calls cuBLAS 12 and cuDNN 9 through CUDA 12 bindings. Its loader (cudarc's) tries the unversioned names `libcublas.so` and `libcudnn.so` before the versioned `libcublas.so.12` and `libcudnn.so.9`, and nothing in volkit can change that order. So when the loader path holds a CUDA 13 toolkit's `lib64` (its unversioned `libcublas.so` points at cuBLAS 13), volkit loads cuBLAS 13 and calls it through 12.x bindings. That ran correctly in testing, but it is not a supported setup. `LoadReport::cublas` reports the version that loaded: anything outside 120000-129999 is this case.

To keep the 12.x libraries, keep CUDA 13 library directories off `LD_LIBRARY_PATH` (and out of `ld.so.conf`) for the process, or put first on `LD_LIBRARY_PATH` a directory holding `libcublas.so -> libcublas.so.12` and `libcudnn.so -> libcudnn.so.9` symlinks to the libraries you want. cuBLAS loads its own `libcublasLt.so.12` by its versioned name. Windows is not affected: CUDA installs its DLLs only under names that carry their major (`cublas64_13.dll`), which the 12.x candidate list never names.

## Layout

| Path | What |
| --- | --- |
| `volkit/` | The crate: `nnunet` (export contract, AFV v2 engine, torch-free export), `pth` (the checkpoint reader), `weights`, `unet3d` (layer program), `dnn` (cuDNN algorithm choice and its on-disk cache), and the private device layer (buffers, checked kernel launches, cuDNN conv plans, cuBLAS GEMMs) |
| `volkit/ptx/` | The embedded PTX tiers and cubins, built from `volkit-kernels` |
| `volkit-kernels/` | cuda-oxide kernel sources, on their own pinned toolchain (outside the workspace) |
| `volkit/src/metal/` | The Metal backend: device layer, MSL kernels (direct tensor-op convolutions, Winograd F(4,3) for levels 0 to 2, InstanceNorm and blend glue) and the AFV v2 forward |
| `scripts/` | PTX and cubin build and staleness check, the PTX comparison against inkrs, and the inkrs reference run for GPU checks |

[docs/KERNELS.md](https://github.com/praveenperera/volkit/blob/master/docs/KERNELS.md) covers the kernel tiers, how to rebuild them, and the entry-by-entry check of the PTX against inkrs's.

## Checks

```sh
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
scripts/check-ptx.sh
```

The tests that read model files skip unless `VOLKIT_AFV2_EXPORT` (an `afv-v2` `.safetensors` export) and `VOLKIT_AFV2_MODEL` (the nnU-Net model folder) name them. The Metal kernel tests touch the GPU, so they skip unless `VOLKIT_METAL_TESTS=1` is set; each checks a kernel against a float64 reference. `examples/afv2_smoke.rs` with `scripts/afv2_inkrs_reference.py` runs one column of real CT through volkit and through inkrs and compares the outputs byte for byte.

## Licence

Licensed under either of MIT or Apache-2.0, at your option. [THIRD_PARTY_NOTICES](https://github.com/praveenperera/volkit/blob/master/THIRD_PARTY_NOTICES) covers the ported villa code, the libdevice functions compiled into the PTX, and the cuda-oxide device crate.
