// Winograd F(4x4x4, 3x3x3) for the stride-1 3x3x3 convs of the mid levels: each 4^3 block of outputs comes from a
// 6^3 block of padded input, transformed (V = B^T d B per axis), multiplied point by point with the host-transformed
// weights (U = G g G^T per axis) as 216 batched GEMMs on matmul2d, then transformed back (Y = A^T M A per axis). That
// takes 216 multiplies per 64 outputs where the direct conv takes 27 per output, 8x fewer. The matrices are Lavin and
// Gray's F(4, 3) with points 0, +-1, +-2, infinity; the transforms run in float32 with every product and sum rounded
// on its own (FP contraction is off for the library)
//
// Layouts: activations are the padded channels-last layout of common.metal; V is [216, tiles, cv] and M is [216,
// tiles, o] for one chunk of tiles (whole planes of tiles in z, so a level's tiles go in a few chunks when its
// scratch is small); U is [216, o, cv] per conv. A tile t of a chunk starting at tile t0 is global tile t0 + t, and
// global tile (tz, ty, tx) covers outputs [4 tz, 4 tz + 4) and so on, its input window padded voxels [4 tz, 4 tz + 6).
// Interior dims must be multiples of 4

// B^T applied in place to six values s apart
static inline void wino_bt(thread float* v, int s) {
    float d0 = v[0], d1 = v[s], d2 = v[2 * s], d3 = v[3 * s], d4 = v[4 * s], d5 = v[5 * s];
    v[0] = 4.0f * d0 - 5.0f * d2 + d4;
    v[s] = -4.0f * d1 - 4.0f * d2 + d3 + d4;
    v[2 * s] = 4.0f * d1 - 4.0f * d2 - d3 + d4;
    v[3 * s] = -2.0f * d1 - d2 + 2.0f * d3 + d4;
    v[4 * s] = 2.0f * d1 - d2 - 2.0f * d3 + d4;
    v[5 * s] = 4.0f * d1 - 5.0f * d3 + d5;
}

// A^T of six values s apart, into four values t apart
static inline void wino_at(thread const float* m, int s, thread float* o, int t) {
    float m0 = m[0], m1 = m[s], m2 = m[2 * s], m3 = m[3 * s], m4 = m[4 * s], m5 = m[5 * s];
    o[0] = m0 + m1 + m2 + m3 + m4;
    o[t] = m1 - m2 + 2.0f * m3 - 2.0f * m4;
    o[2 * t] = m1 + m2 + 4.0f * m3 + 4.0f * m4;
    o[3 * t] = m1 - m2 + 8.0f * m3 - 8.0f * m4 + m5;
}

// Channels per threadgroup in the transforms; 6 threads per channel, one per z plane of the 6^3 block
constant uint WINO_CH = 32;

// The input transform of channels [32 g, 32 g + 32) of x (c channels, padded [d + 2, h + 2, w + 2, c]) for chunk
// tile tg.x, into V's channels [coff, coff + c) of cv. Thread (ch, z) transforms plane z of the block along x and y
// into threadgroup memory, then transforms six of the 36 (y, x) columns along z. Grid (tiles in chunk, ceil(c / 32)),
// 192 threads
kernel void wino_in(device float* p_v [[buffer(0)]], device const float* p_x [[buffer(1)]],
                    constant uint& d [[buffer(2)]], constant uint& h [[buffer(3)]], constant uint& w [[buffer(4)]],
                    constant uint& c [[buffer(5)]], constant uint& cv [[buffer(6)]], constant uint& coff [[buffer(7)]],
                    constant uint& t0 [[buffer(8)]], constant uint& tiles [[buffer(9)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float plane[6 * 36 * WINO_CH];
    uint cl = tid % WINO_CH, z = tid / WINO_CH;
    uint ch = tg.y * WINO_CH + cl;
    uint nty = h / 4, ntx = w / 4;
    uint t = t0 + tg.x;
    uint tz = t / (nty * ntx), ty = (t / ntx) % nty, tx = t % ntx;
    uint wp = w + 2, hp = h + 2;
    if (ch < c) {
        float a[36];
        for (uint y = 0; y < 6; y++) {
            ulong row = ((ulong)(4 * tz + z) * hp + 4 * ty + y) * wp + 4 * tx;
            for (uint x = 0; x < 6; x++) a[y * 6 + x] = p_x[(row + x) * c + ch];
        }
        for (uint y = 0; y < 6; y++) wino_bt(a + y * 6, 1);
        for (uint x = 0; x < 6; x++) wino_bt(a + x, 6);
        for (uint i = 0; i < 36; i++) plane[(z * 36 + i) * WINO_CH + cl] = a[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ch >= c) return;
    for (uint k = 0; k < 6; k++) {
        uint i = z * 6 + k;
        float col[6];
        for (uint pz = 0; pz < 6; pz++) col[pz] = plane[(pz * 36 + i) * WINO_CH + cl];
        wino_bt(col, 1);
        for (uint pz = 0; pz < 6; pz++) p_v[((ulong)(pz * 36 + i) * tiles + tg.x) * cv + coff + ch] = col[pz];
    }
}

// The output transform of M's channels [32 g, 32 g + 32) of o for chunk tile tg.x, plus the conv's bias, into the
// interior of the padded y [d + 2, h + 2, w + 2, o]. Thread (ch, z) takes plane z of the 6^3 block along x and y
// (6 x 6 to 4 x 4), then four of the threads per channel take four of the 16 (y, x) columns each along z. Grid (tiles
// in chunk, ceil(o / 32)), 192 threads
kernel void wino_out(device float* p_y [[buffer(0)]], device const float* p_m [[buffer(1)]],
                     device const float* p_bias [[buffer(2)]], constant uint& d [[buffer(3)]],
                     constant uint& h [[buffer(4)]], constant uint& w [[buffer(5)]], constant uint& o [[buffer(6)]],
                     constant uint& t0 [[buffer(7)]], constant uint& tiles [[buffer(8)]],
                     constant uint& flags [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float plane[6 * 16 * WINO_CH];
    uint cl = tid % WINO_CH, z = tid / WINO_CH;
    uint ch = tg.y * WINO_CH + cl;
    uint nty = h / 4, ntx = w / 4;
    uint t = t0 + tg.x;
    uint tz = t / (nty * ntx), ty = (t / ntx) % nty, tx = t % ntx;
    if (ch < o) {
        float m[36];
        for (uint i = 0; i < 36; i++) m[i] = p_m[((ulong)(z * 36 + i) * tiles + tg.x) * o + ch];
        // along x: six rows of six into six rows of four
        float r[24];
        for (uint y = 0; y < 6; y++) wino_at(m + y * 6, 1, r + y * 4, 1);
        // along y: four columns of six into a 4 x 4 block
        float q[16];
        for (uint x = 0; x < 4; x++) wino_at(r + x, 4, q + x, 4);
        for (uint i = 0; i < 16; i++) plane[(z * 16 + i) * WINO_CH + cl] = q[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ch >= o || z >= 4) return;
    float b = (flags & CONV_BIAS) ? p_bias[ch] : 0.0f;
    for (uint k = 0; k < 4; k++) {
        uint i = z * 4 + k;
        float col[6], out[4];
        for (uint pz = 0; pz < 6; pz++) col[pz] = plane[(pz * 16 + i) * WINO_CH + cl];
        wino_at(col, 1, out, 1);
        uint oy = i / 4, ox = i % 4;
        for (uint oz = 0; oz < 4; oz++) {
            float v = (flags & CONV_BIAS) ? out[oz] + b : out[oz];
            p_y[(ulong)prow(4 * tz + oz, 4 * ty + oy, 4 * tx + ox, h, w) * o + ch] = v;
        }
    }
}

// M[p] = V[p] . U[p]^T for the 216 points p: V[p] [m, k], U[p] [n, k], M[p] [m, n]; grid (n tiles, m tiles, 216),
// 128 threads
template <int BM, int BN>
kernel void wino_gemm(device float* p_v [[buffer(0)]], device float* p_u [[buffer(1)]], device float* p_m [[buffer(2)]],
                      constant uint& m [[buffer(3)]], constant uint& n [[buffer(4)]], constant uint& k [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
    int m0 = tg.y * BM, n0 = tg.x * BN;
    ulong p = tg.z;
    mat<float> tx(p_v + p * m * k, dextents<int32_t, 2>(k, m), array<int, 2>{1, (int)k});
    mat<float> tw(p_u + p * n * k, dextents<int32_t, 2>(k, n), array<int, 2>{1, (int)k});
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mx = tx.slice(0, m0);
    auto mw = tw.slice(0, n0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mx), decltype(mw), float>();
    op.run(mx, mw, acc);
    device float* y = p_m + p * m * n;
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0], row = m0 + idx[1];
        if (row < (int)m && col < (int)n) y[(ulong)row * n + col] = acc[i];
    }
}

#define WINO_GEMM(BM, BN)                                                                                              \
    template [[host_name("wino_gemm_" #BM "x" #BN)]] kernel void wino_gemm<BM, BN>(                                 \
        device float*, device float*, device float*, constant uint&, constant uint&, constant uint&, uint3);
WINO_GEMM(64, 64)
WINO_GEMM(64, 32)
WINO_GEMM(32, 64)

// Zeroes every row of a padded buffer [d + 2, h + 2, w + 2, c] followed by slack rows (`rows` in all) that is not an
// interior voxel, after the buffer served as Winograd scratch, so the zero padding the tap convs rely on holds again;
// the interior is left as it is, since every producer rewrites it whole. Border row b runs over the z = 0 and
// z = d + 1 planes, then each interior plane's y = 0 and y = h + 1 lines and its x = 0 and x = w + 1 voxels, then
// the slack. One thread per (border row, channel)
kernel void zero_border(device float* p_x [[buffer(0)]], constant uint& d [[buffer(1)]], constant uint& h [[buffer(2)]],
                        constant uint& w [[buffer(3)]], constant uint& c [[buffer(4)]],
                        constant uint& rows [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    uint wp = w + 2, hp = h + 2, plane = hp * wp;
    uint per_plane = 2 * wp + 2 * h;
    uint body = (d + 2) * plane;
    uint count = 2 * plane + d * per_plane + (rows - body);
    if (i >= count * c) return;
    uint b = i / c, ch = i % c;
    uint r;
    if (b < 2 * plane) {
        r = b < plane ? b : (d + 1) * plane + (b - plane);
    } else if (b < 2 * plane + d * per_plane) {
        uint q = b - 2 * plane;
        uint z = 1 + q / per_plane;
        q %= per_plane;
        uint y, x;
        if (q < wp) {
            y = 0, x = q;
        } else if (q < 2 * wp) {
            y = h + 1, x = q - wp;
        } else {
            q -= 2 * wp;
            y = 1 + q / 2, x = (q % 2) ? w + 1 : 0;
        }
        r = (z * hp + y) * wp + x;
    } else {
        r = body + (b - 2 * plane - d * per_plane);
    }
    p_x[(ulong)r * c + ch] = 0.0f;
}
