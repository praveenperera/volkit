// Shared header of the volkit Metal kernels, compiled first. Pointer parameters are named `p_*`: the host's argument
// check reads the names to tell a pointer from a `constant uint&` scalar
//
// Activations are float32, channels-last, at batch 1. Every level's activations live in a zero-padded layout
// [d + 2, h + 2, w + 2, c] followed by slack rows (`padded_rows` on the host), so a 3x3x3 conv reads its input with no
// bounds checks and no separate pad pass: producers write interiors only, and the host zeroes each padded buffer once,
// so its borders stay zero. Products and sums the CUDA kernels round separately stay separate, so FP contraction is
// off for the whole library, as it is in inkrs's
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#pragma METAL fp contract(off)

template <typename T>
using mat = tensor<device T, dextents<int32_t, 2>, tensor_inline>;

// Padded row of interior voxel (z, y, x) of a [d, h, w] activation
static inline uint prow(uint z, uint y, uint x, uint h, uint w) {
    return ((z + 1) * (h + 2) + y + 1) * (w + 2) + x + 1;
}
