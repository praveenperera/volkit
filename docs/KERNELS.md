# Kernels

The GPU kernels are written in Rust with [cuda-oxide](https://github.com/NVIDIA/cuda-rust) in `volkit-kernels/`,
compiled offline to PTX and embedded in the `volkit` crate (`volkit/ptx/`), so building volkit needs neither
cuda-oxide nor a CUDA toolkit.

## Tiers

| Tier | PTX target and ISA | Loaded on | Cubins (SASS) shipped |
| --- | --- | --- | --- |
| sm80 | `sm_80`, ISA 7.0 | compute capability 8.0 to 11.x | sm_80, sm_86, sm_89, sm_90, sm_100 |
| sm120 | `sm_120`, ISA 8.7 | 12.x (RTX 50) | sm_120 |

A GPU with a cubin for its family loads it; any other GPU of 8.0 or above JIT-compiles the tier's PTX on first
use. `VOLKIT_PTX_TIER=sm80` forces the sm80 tier on an RTX 50. GPUs below compute capability 8.0 are refused
as unsupported: the 3D U-Net glue rounds through bf16 for cuDNN's bf16 convolutions, which need 8.0.

## Building

```sh
scripts/build-ptx.sh    # cuda-oxide (pinned toolchain, Rust 1.99.1) + CUDA toolkit: PTX, cubins, manifest
scripts/build-cubin.sh  # CUDA toolkit only: cubins from the committed PTX, manifest
scripts/check-ptx.sh    # no toolkit: the committed PTX and cubins match their sources (CI runs this)
```

`volkit-kernels` pins cuda-oxide to NVIDIA/cuda-rust `8d825486993f806903d8968232214ddafa8ed7da` and the
toolchain cuda-oxide pins (`volkit-kernels/rust-toolchain.toml`). `volkit/ptx/tools.txt` records the LLVM tools
and libdevice the PTX was built with, `volkit/ptx/ptxas.txt` the ptxas that compiled the cubins (CUDA 13.2).

## Provenance and the check against inkrs

The kernels are inkrs's (`inkrs-kernels` at inkrs commit `c1a9850`, release 0.4.0 with the AfvV2 bf16 opt-in),
reduced to the ones the AFV v2 engine launches:

- `unet3d`: the 3D U-Net glue in bf16 (`u3_*`), fp16 (`u3h_*`) and float32 (`u3f_*`), and `u3_finalize`
- `afv2`: `af_prep`, `af_fold`, `af_finish`, `af_pool`
- `nnunet::nn_prep` and `fiber::nf_head`, the input and seg head of AfvV2's bf16 mode
- `common`: the libdevice math and FP16 conversions those kernels call

Nothing a kept kernel computes was edited; the removed kernels, inkrs's Turing (sm_75) fallbacks and the helpers
only removed kernels used were deleted around them.

Built with the same cuda-oxide commit, toolchain, LLVM, libdevice and ptxas as inkrs's committed PTX, every kept
kernel's `.entry` (signature and body) is identical to inkrs `c1a9850`'s, in both tiers:

```sh
python3 -I scripts/compare-ptx-entries.py volkit/ptx/volkit.sm80.ptx  <inkrs>/ptx/inkrs.sm80.ptx
python3 -I scripts/compare-ptx-entries.py volkit/ptx/volkit.sm120.ptx <inkrs>/ptx/inkrs.sm120.ptx
```

| Tier | Entries compared | Identical | Register counts (ptxas `-v`) |
| --- | --- | --- | --- |
| sm80 | 28 | 28 | identical for all 28 |
| sm120 | 28 | 28 | identical for all 28 |

The comparison ignores only what names an entry's place in its module: block labels (`$L__BB<index>_`, the
entry's index in the file), shared-array symbols (a hash of the defining crate; their declarations are compared
by type, alignment and size instead) and comments.
