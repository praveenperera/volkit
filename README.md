# volkit

Shared native GPU runtime for 3D volume networks: the CUDA host layer over cudakit, cuDNN and cuBLAS layers, PTX kernels, tiling and blending, zarr IO, and the networks inkrs and fibrs run on it.

It currently holds what native AFV fibre model V2 inference needs: the AFV v2 engine (`volkit::nnunet::afv2::AfvV2`), the 3D ResEnc U-Net forward it runs, and a reader for nnU-Net `checkpoint_final.pth` files that needs no torch or Python. The code and kernels came from [inkrs](https://github.com/praveenperera/inkrs) 0.4.0's AFV v2 path and compute the same output.

## Use

```toml
[target.'cfg(target_os = "linux")'.dependencies]
volkit = "0.1"
```

```rust
use volkit::dnn::Pick;
use volkit::nnunet::afv2::{AfvPrecision, AfvV2, Options};

let opts = Options { device: 0, pick: Pick::Find, precision: AfvPrecision::Fp32 };
let mut net = AfvV2::open(model_dir_or_export, &opts)?;
net.begin([nz, ny, nx], y0, x0)?;
net.add(&tiles, &origins)?; // up to MAX_BATCH 128^3 uint8 tiles at their padded lattice origins
let column = net.finish(true)?; // semantic (2, nz, ny, nx) and tracing (2, nz/2, ny/2, nx/2) uint8
```

[docs/API.md](https://github.com/praveenperera/volkit/blob/master/docs/API.md) lists the public API and environment variables.

## Requirements

- Linux or Windows x86_64 with an NVIDIA GPU of compute capability 8.0 or above (RTX 30 / A100 and later). On macOS the crate builds empty.
- At run time: the NVIDIA driver, cuDNN 9 and cuBLAS, loaded with `dlopen` and checked before first use. Building needs no CUDA toolkit.

## Layout

| Path | What |
| --- | --- |
| `volkit/` | The crate: `gpu` (device, buffers, checked kernel launches), `dnn` (cuDNN conv plans, cuBLAS GEMMs, the on-disk algorithm cache), `unet3d` (layer program and forward), `nnunet` (export contract, AFV v2 engine, torch-free export), `weights`, `zarr` |
| `volkit/ptx/` | The embedded PTX tiers and cubins, built from `volkit-kernels` |
| `volkit-kernels/` | cuda-oxide kernel sources, on their own pinned toolchain (outside the workspace) |
| `scripts/` | PTX and cubin build and staleness check, the PTX comparison against inkrs, and the inkrs reference run for GPU checks |

[docs/KERNELS.md](https://github.com/praveenperera/volkit/blob/master/docs/KERNELS.md) covers the kernel tiers, how to rebuild them, and the entry-by-entry check of the PTX against inkrs's.

## Checks

```sh
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
scripts/check-ptx.sh
```

The tests that read model files skip unless `VOLKIT_AFV2_EXPORT` (an `afv-v2` `.safetensors` export) and `VOLKIT_AFV2_MODEL` (the nnU-Net model folder) name them. `examples/afv2_smoke.rs` with `scripts/afv2_inkrs_reference.py` runs one column of real CT through volkit and through inkrs and compares the outputs byte for byte.

## Licence

Licensed under either of MIT or Apache-2.0, at your option. [THIRD_PARTY_NOTICES](https://github.com/praveenperera/volkit/blob/master/THIRD_PARTY_NOTICES) covers the ported villa code, the libdevice functions compiled into the PTX, and the cuda-oxide device crate.
