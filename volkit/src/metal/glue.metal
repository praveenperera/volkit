// The AFV v2 forward's glue on Metal: the stem, InstanceNorm, residuals, LeakyReLU, the skip pools, the seg head's
// softmax and blend, and the column's finalisation. These are the MSL ports of volkit-kernels' `unet3d::fp32`
// (`u3f_*`, `u3_finalize`) and `afv2` (`af_*`) kernels, in the same float32 arithmetic and order: Welford statistics
// merged by Chan's rule, the conv bias added on its own, `fma(gamma * (x - mean), invstd, beta)`, the residual add,
// then LeakyReLU, and the reference's blend with each product and sum rounded separately
//
// Buffers carry no length on Metal, so every kernel takes the element counts it bounds-checks against

// ---------------------------------------------------------------- small helpers

kernel void fill_u32(device uint* p_x [[buffer(0)]], constant uint& value [[buffer(1)]], constant uint& count [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < count) p_x[i] = value;
}

// Chan's merge of two (n, mean, m2) Welford states, as volkit-kernels' `chan`
static inline float3 chan(float3 a, float3 b) {
    if (b.x == 0.0f) return a;
    if (a.x == 0.0f) return b;
    float n = a.x + b.x;
    float wgt = b.x / n;
    float e = b.y - a.y;
    return float3(n, a.y + e * wgt, a.z + b.z + e * e * a.x * wgt);
}

// The interior voxel `r` of a [d, h, w] activation as a row of its layout: padded, or plain when `padded` is 0
static inline uint layout_row(uint r, uint h, uint w, uint padded) {
    if (padded == 0) return r;
    uint x = r % w, q = r / w;
    return prow(q / h, q % h, x, h, w);
}

// ---------------------------------------------------------------- stem

// The stem conv, 1 -> o channels, 3x3x3, pad 1, straight from the uint8 tile: each code goes through the tile's
// float32 normalisation table (the reference's `(code - mean) / max(std, 10)`), voxels past the tile edge read the
// conv's zero padding. One thread per (voxel, group of 8 output channels); weights [o, 27]; the raw output plus the
// bias goes to the padded y [p + 2]^3 x o
kernel void stem(device float* p_y [[buffer(0)]], device const uchar* p_tile [[buffer(1)]],
                 device const float* p_table [[buffer(2)]], device const float* p_w [[buffer(3)]],
                 device const float* p_bias [[buffer(4)]], constant uint& p [[buffer(5)]], constant uint& o [[buffer(6)]],
                 constant uint& count [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint groups = o / 8;
    uint g = i % groups, v = i / groups;
    uint x = v % p, r = v / p;
    uint y = r % p, z = r / p;
    float acc[8];
    for (uint k = 0; k < 8; k++) acc[k] = 0.0f;
    for (uint t = 0; t < 27; t++) {
        int zz = int(z) + int(t / 9) - 1, yy = int(y) + int((t / 3) % 3) - 1, xx = int(x) + int(t % 3) - 1;
        if (zz < 0 || zz >= int(p) || yy < 0 || yy >= int(p) || xx < 0 || xx >= int(p)) continue;
        float xv = p_table[p_tile[((uint)zz * p + (uint)yy) * p + (uint)xx]];
        for (uint k = 0; k < 8; k++) acc[k] = fma(xv, p_w[(g * 8 + k) * 27 + t], acc[k]);
    }
    ulong out = (ulong)prow(z, y, x, p, p) * o + g * 8;
    for (uint k = 0; k < 8; k++) p_y[out + k] = acc[k] + p_bias[g * 8 + k];
}

// ---------------------------------------------------------------- InstanceNorm

// Welford partial statistics of one chunk of lines (z, y) of a [d, h, w, c] activation (padded layout, or plain when
// `padded` is 0), four channels a thread: partial[chunk, ch] = (n, mean, m2). Grid (chunks, ceil(c / 4 / qpg)),
// qpg * lanes threads: thread t owns channel quad g qpg + t % qpg and voxels x = t / qpg, + lanes, ... of each of the
// chunk's lines, so a line's quads load as consecutive float4s; lane 0 then merges the lanes in order. The four
// channels share the count, so a voxel costs one reciprocal. c must be a multiple of 4
kernel void in_partial(device float4* p_part [[buffer(0)]], device const float4* p_x [[buffer(1)]],
                       constant uint& d [[buffer(2)]], constant uint& h [[buffer(3)]], constant uint& w [[buffer(4)]],
                       constant uint& c [[buffer(5)]], constant uint& padded [[buffer(6)]],
                       constant uint& chunk_lines [[buffer(7)]], constant uint& qpg [[buffer(8)]],
                       constant uint& lanes [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sn[256];
    threadgroup float4 smean[256];
    threadgroup float4 sm2[256];
    uint cq = c / 4;
    uint q = tg.y * qpg + tid % qpg, lane = tid / qpg;
    uint l0 = tg.x * chunk_lines, l1 = min(l0 + chunk_lines, d * h);
    float n = 0.0f;
    float4 mean = float4(0.0f), m2 = float4(0.0f);
    if (q < cq) {
        for (uint l = l0; l < l1; l++) {
            ulong base = padded ? (ulong)prow(l / h, l % h, 0, h, w) : (ulong)l * w;
            for (uint x = lane; x < w; x += lanes) {
                float4 v = p_x[(base + x) * cq + q];
                n += 1.0f;
                float inv = 1.0f / n;
                float4 dlt = v - mean;
                mean += dlt * inv;
                m2 += dlt * (v - mean);
            }
        }
    }
    sn[tid] = n;
    smean[tid] = mean;
    sm2[tid] = m2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane != 0 || q >= cq) return;
    for (uint l = 1; l < lanes; l++) {
        uint o = l * qpg + tid;
        float nb = sn[o];
        if (nb == 0.0f) continue;
        if (n == 0.0f) {
            n = nb, mean = smean[o], m2 = sm2[o];
            continue;
        }
        float nn = n + nb;
        float wgt = nb / nn;
        float4 e = smean[o] - mean;
        mean = mean + e * wgt;
        m2 = m2 + sm2[o] + e * e * n * wgt;
        n = nn;
    }
    ulong out = (ulong)tg.x * c + 4 * q;
    for (uint k = 0; k < 4; k++) p_part[out + k] = float4(n, mean[k], m2[k], 0.0f);
}

// Per channel (mean, invstd) from in_partial's chunks: thread t merges chunks t, t + 256, ... in order, then a fixed
// tree merges the threads, so the result is deterministic. invstd = 1 / sqrt(var + eps) with the biased variance, as
// u3_finalize. Grid c threadgroups of 256
kernel void in_finalize(device float2* p_stats [[buffer(0)]], device const float4* p_part [[buffer(1)]],
                        constant uint& chunks [[buffer(2)]], constant uint& c [[buffer(3)]],
                        constant float& eps [[buffer(4)]], uint ch [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float3 acc[256];
    float3 st = float3(0.0f);
    for (uint k = tid; k < chunks; k += 256) st = chan(st, p_part[(ulong)k * c + ch].xyz);
    acc[tid] = st;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half_ = 128; half_ > 0; half_ /= 2) {
        if (tid < half_) acc[tid] = chan(acc[tid], acc[tid + half_]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float3 a = acc[0];
        float var = a.x > 0.0f ? a.z / a.x : 0.0f;
        var = var > 0.0f ? var : 0.0f;
        p_stats[ch] = float2(a.y, 1.0f / sqrt(var + eps));
    }
}

// ATen's InstanceNorm transform, the trailing multiply-add fused as nvcc compiles it
static inline float in_affine(float v, float2 s, float gamma, float beta) {
    return fma(gamma * (v - s.x), s.y, beta);
}

constant uint RES_NONE = 0;
constant uint RES_PLAIN = 1;  // a normalised activation of the output's shape, in the padded layout
constant uint RES_NORMED = 2; // a bias-free 1x1 conv output, plain, normalised on load by its own statistics
constant uint RES_POOLED = 3; // an average-pooled skip of the output's shape, plain, added as it is

// InstanceNorm apply in place on the padded conv output y, with an optional residual and LeakyReLU:
// y = leaky?(affine(y) + residual?), as u3f_apply. One thread per (voxel, 4 channels)
kernel void in_apply(device float4* p_y [[buffer(0)]], device const float2* p_stats [[buffer(1)]],
                     device const float* p_w [[buffer(2)]], device const float* p_b [[buffer(3)]],
                     device const float4* p_res [[buffer(4)]], device const float2* p_rstats [[buffer(5)]],
                     device const float* p_rw [[buffer(6)]], device const float* p_rb [[buffer(7)]],
                     constant uint& d [[buffer(8)]], constant uint& h [[buffer(9)]], constant uint& w [[buffer(10)]],
                     constant uint& c [[buffer(11)]], constant uint& res [[buffer(12)]], constant uint& act [[buffer(13)]],
                     constant float& slope [[buffer(14)]], uint i [[thread_position_in_grid]]) {
    uint quads = c / 4;
    uint rows = d * h * w;
    if (i >= rows * quads) return;
    uint q = i % quads, r = i / quads;
    uint ch = 4 * q;
    ulong o = (ulong)layout_row(r, h, w, 1) * quads + q;
    float4 v = p_y[o];
    float4 out;
    for (uint k = 0; k < 4; k++) out[k] = in_affine(v[k], p_stats[ch + k], p_w[ch + k], p_b[ch + k]);
    if (res == RES_PLAIN) {
        out = out + p_res[o];
    } else if (res == RES_POOLED) {
        out = out + p_res[(ulong)r * quads + q];
    } else if (res == RES_NORMED) {
        float4 s = p_res[(ulong)r * quads + q];
        for (uint k = 0; k < 4; k++) out[k] = out[k] + in_affine(s[k], p_rstats[ch + k], p_rw[ch + k], p_rb[ch + k]);
    }
    if (act != 0) {
        for (uint k = 0; k < 4; k++) out[k] = out[k] > 0.0f ? out[k] : out[k] * slope;
    }
    p_y[o] = out;
}

// AvgPool3d with kernel = stride (fz, fy, fx) of the padded x [fz d, fy h, fx w, c] into the plain [d, h, w, c]: the
// float32 sum in ATen's order (z, then y, then x) times 1 / n, exact as n is a power of two. One thread per element
kernel void avgpool(device float* p_out [[buffer(0)]], device const float* p_x [[buffer(1)]],
                    constant uint& d [[buffer(2)]], constant uint& h [[buffer(3)]], constant uint& w [[buffer(4)]],
                    constant uint& c [[buffer(5)]], constant uint& fz [[buffer(6)]], constant uint& fy [[buffer(7)]],
                    constant uint& fx [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i >= d * h * w * c) return;
    uint ch = i % c, pix = i / c;
    uint x = pix % w, r = pix / w;
    uint y = r % h, z = r / h;
    uint hi = fy * h, wi = fx * w;
    float s = 0.0f;
    for (uint a = 0; a < fz; a++)
        for (uint b = 0; b < fy; b++)
            for (uint e = 0; e < fx; e++) s = s + p_x[(ulong)prow(fz * z + a, fy * y + b, fx * x + e, hi, wi) * c + ch];
    p_out[i] = s * (1.0f / float(fz * fy * fx));
}

// ---------------------------------------------------------------- AFV v2 head, blend and finalisation

// The seg head and the blend of one tile into the column, as the reference's 1x1x1 conv, its separate bias add,
// `softmax(1)[:, 1:3]`, then `acc[:, d] += p[:, s] * w` and `wsum[d] += w` (af_fold): the four logits are float32 dot
// products of the last decoder output x (padded [p + 2]^3 x cin) with the head's weights [4, cin], plus the bias;
// the softmax takes the classes' maximum, exp(l - max) summed in class order, then each exp / sum; the tile weight
// is (ramp[z] * ramp[y]) * ramp[x]. The tile voxels [s, s + e) per axis land on column voxels [dd, dd + e); the column
// is (nz, ny, nx) with n voxels, acc holding the vertical channel then the horizontal one. One thread per voxel of
// the extent; tiles are folded one dispatch at a time in the column's order
kernel void head_fold(device float* p_acc [[buffer(0)]], device float* p_wsum [[buffer(1)]],
                      device const float* p_x [[buffer(2)]], device const float* p_hw [[buffer(3)]],
                      device const float* p_hb [[buffer(4)]], device const float* p_ramp [[buffer(5)]],
                      constant uint& p [[buffer(6)]], constant uint& cin [[buffer(7)]], constant uint& sz [[buffer(8)]],
                      constant uint& sy [[buffer(9)]], constant uint& sx [[buffer(10)]], constant uint& ez [[buffer(11)]],
                      constant uint& ey [[buffer(12)]], constant uint& ex [[buffer(13)]],
                      constant uint& dz [[buffer(14)]], constant uint& dy [[buffer(15)]],
                      constant uint& dx [[buffer(16)]], constant uint& ny [[buffer(17)]],
                      constant uint& nx [[buffer(18)]], constant uint& n [[buffer(19)]],
                      uint e [[thread_position_in_grid]]) {
    if (e >= ez * ey * ex) return;
    uint ix = e % ex, iy = (e / ex) % ey, iz = e / (ex * ey);
    uint tz = sz + iz, ty = sy + iy, tx = sx + ix;
    uint o = ((dz + iz) * ny + dy + iy) * nx + dx + ix;
    if (tz >= p || ty >= p || tx >= p || o >= n) return;
    device const float* xv = p_x + (ulong)prow(tz, ty, tx, p, p) * cin;
    float l[4];
    for (uint k = 0; k < 4; k++) {
        float s = 0.0f;
        for (uint ch = 0; ch < cin; ch++) s = fma(xv[ch], p_hw[k * cin + ch], s);
        l[k] = s + p_hb[k];
    }
    float m = max(max(l[0], l[1]), max(l[2], l[3]));
    float ex0 = exp(l[0] - m), ex1 = exp(l[1] - m), ex2 = exp(l[2] - m), ex3 = exp(l[3] - m);
    float s = ((ex0 + ex1) + ex2) + ex3;
    float pv = ex1 / s, ph = ex2 / s;
    float wt = (p_ramp[tz] * p_ramp[ty]) * p_ramp[tx];
    p_acc[o] = p_acc[o] + pv * wt;
    p_acc[n + o] = p_acc[n + o] + ph * wt;
    p_wsum[o] = p_wsum[o] + wt;
}

constant uint AF_NONFINITE = 1;

static inline bool finite_f(float v) { return fabs(v) <= 3.4028235e38f; }

// The reference `quantize`: rint(clamp(p, 0, 1) * 255), round half to even
static inline uchar quantize(float v) { return uchar(rint(min(max(v, 0.0f), 1.0f) * 255.0f)); }

// The column's normalisation and `semantic` output, as af_finish: p = acc / wsum where wsum > 0 and acc / 1
// elsewhere, in place in acc, then quantize into sem ([2, n]); uncovered voxels are counted into stat[1] and a
// non-finite p sets AF_NONFINITE in stat[0]. Threadgroups of 256 over the n voxels
kernel void af_finish(device uchar* p_sem [[buffer(0)]], device float* p_acc [[buffer(1)]],
                      device atomic_uint* p_stat [[buffer(2)]], device const float* p_wsum [[buffer(3)]],
                      constant uint& n [[buffer(4)]], uint e [[thread_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint tally;
    if (tid == 0) atomic_store_explicit(&tally, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e < n) {
        float c = p_wsum[e];
        bool covered = c > 0.0f;
        if (!covered) atomic_fetch_add_explicit(&tally, 1u, memory_order_relaxed);
        float dv = covered ? c : 1.0f;
        bool bad = false;
        for (uint k = 0; k < 2; k++) {
            ulong o = (ulong)k * n + e;
            float v = p_acc[o] / dv;
            p_acc[o] = v;
            bad = bad || !finite_f(v);
            p_sem[o] = quantize(v);
        }
        if (bad) atomic_fetch_or_explicit(&p_stat[0], AF_NONFINITE, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint count = atomic_load_explicit(&tally, memory_order_relaxed);
        if (count != 0) atomic_fetch_add_explicit(&p_stat[1], count, memory_order_relaxed);
    }
}

// The column's `tracing` output, as af_pool: per channel the float32 sum from zero of each 2x2x2 block of af_finish's
// p in (z, y, x fastest) order, times 1/8, then quantize; odd last planes are left out
kernel void af_pool(device uchar* p_tra [[buffer(0)]], device atomic_uint* p_stat [[buffer(1)]],
                    device const float* p_p [[buffer(2)]], constant uint& nz [[buffer(3)]], constant uint& ny [[buffer(4)]],
                    constant uint& nx [[buffer(5)]], uint e [[thread_position_in_grid]]) {
    uint hz = nz / 2, hy = ny / 2, hx = nx / 2;
    uint hn = hz * hy * hx, n = nz * ny * nx;
    if (e >= 2 * hn) return;
    uint k = e / hn, q = e % hn;
    uint x = q % hx, y = (q / hx) % hy, z = q / (hx * hy);
    float s = 0.0f;
    for (uint a = 0; a < 8; a++) {
        uint oz = a >> 2, oy = (a >> 1) & 1, ox = a & 1;
        s = s + p_p[(ulong)k * n + ((2 * z + oz) * ny + 2 * y + oy) * nx + 2 * x + ox];
    }
    float v = s * 0.125f;
    if (!finite_f(v)) atomic_fetch_or_explicit(&p_stat[0], AF_NONFINITE, memory_order_relaxed);
    p_tra[e] = quantize(v);
}
