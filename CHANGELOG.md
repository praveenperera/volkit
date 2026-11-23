# Changelog

All notable changes to volkit are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added

- A native Metal backend for Apple silicon. On macOS, `volkit::nnunet::afv2::AfvV2` runs AFV v2 on the GPU
  through the same API as on CUDA: MSL kernels compiled when the engine opens, so a build needs neither Xcode nor
  the Metal toolchain, with the convolutions on Metal 4 tensor ops (`matmul2d`) in float32 and Winograd F(4,3)
  for the stride-1 convolutions of levels 0 to 2. It needs macOS 26 or newer on an M1 or newer.
  - Accuracy: on real 128³ tiles of PHerc0826, the output matches PyTorch's float32 CPU forward to at most 1/255,
    with under 0.01% of values differing and the same most-likely class on every foreground voxel. A build gives
    the same bytes for the same input on every run.
  - Speed and memory on an M4 Pro, 16 tiles of PHerc0826 timed in alternating pairs: 226.6 ms a tile against
    402.1 ms for PyTorch 2.15 on MPS (1.77×), at a 2.69 GB peak against MPS's 4.90 GB.
  - `Options` there holds only the device (0) and the precision; `AfvPrecision::Bf16` is CUDA-only and fails with
    an unsupported error. `LoadReport` lists the device, the model's device bytes, the recommended working set and
    the Winograd levels. [docs/API.md](docs/API.md#metal) has the details and the `VOLKIT_METAL_*` variables.

### Changed

- The crate no longer builds empty on macOS, so consumers can depend on it without a target gate.

[0.2.0]: https://github.com/praveenperera/volkit/compare/v0.1.2...v0.2.0

## [0.1.2] - 2026-11-20

### Added

- `AfvV2::finish_into` takes the semantic and tracing vectors of a column the
  caller has finished with and downloads the next column into their
  allocations. A column's outputs are about 520 MB at nz 1000, and a fresh
  allocation that size is paged in again on every column. With the caller
  handing its vectors back, predict on a 9-column band of PHerc. 0826 at fp32
  takes 37.2 s, against 40.8 s without reuse and 42.4 s on 0.1.0, and writes
  byte-identical output. `finish` keeps its signature and allocates as before.

### Changed

- GPU downloads copy into uninitialised memory instead of zero-filling a vector
  and overwriting it. A vector too small for a download is replaced by one of
  exactly the size needed rather than grown, so a 430 MB vector handed back for
  a 461 MB column no longer asks for up to 860 MB.
- `Checkpoint::take` moves a tensor's data out of the checkpoint instead of
  copying it, and loading builds transposed conv weights and encodes stored
  bytes without zero-filled scratch buffers.

[0.1.2]: https://github.com/praveenperera/volkit/compare/v0.1.1...v0.1.2
