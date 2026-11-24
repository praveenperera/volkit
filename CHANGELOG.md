# Changelog

All notable changes to volkit are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The device layer is public, so another runtime can run its own kernels on
  volkit's GPU. `gpu` is a public module, and `Gpu::load_module` loads a
  second PTX module (the highest tier at or below the device's) whose
  kernels launch through `Module::launch` on volkit's stream. Each launch is
  checked against the kernel's PTX parameters, as volkit's own are, and
  `Module::set_max_dynamic_smem` raises a kernel's shared-memory limit. The
  cuDNN and cuBLAS wrappers in `dnn` (`Dnn`, `Blas`, `ConvShape`,
  `ConvPlan`, `Dtype`, `AlgoPerf` and `gemm_bf16`) are public too.
- `Arg::Ptr` passes a raw device pointer to a kernel parameter the PTX marks
  `.ptr`, and `Arg::U64` passes a 64-bit scalar.

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
