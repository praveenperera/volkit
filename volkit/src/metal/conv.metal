// The U-Net's convolutions on Metal 4 tensor ops (matmul2d, float32 operands and accumulation), with no im2col: each
// 3x3x3 conv reads shifted row views of its zero-padded input. The three dx taps of a (dz, dy) are consecutive padded
// rows, and the channels-last weights [o, 27, c] hold them contiguously, so a K = 3c view with row stride c (or 2c
// for a stride-2 conv) covers all three: 9 matmuls a tile instead of 27. The pattern is inkrs's Hecate tap conv
//
// Every conv writes its raw output plus the conv's bias (one float add, as PyTorch adds a conv's bias as its own op)
// into the interior of a padded buffer; the InstanceNorm that follows (glue.metal) normalises it in place

constant uint CONV_BIAS = 1;

// One output element of a conv tile: acc plus the bias where the conv has one
static inline float conv_out(float acc, uint col, device const float* bias, uint flags) {
    return (flags & CONV_BIAS) ? acc + bias[col] : acc;
}

// 3x3x3 conv, stride 1, pad 1: x padded [d + 2, h + 2, w + 2, c] (and `xrows` allocated rows), weights [o, 27, c],
// output y padded [d + 2, h + 2, w + 2, o]. With c2 > 0 a second input x2 (c2 channels, same dims) and weights
// w2 [o, 27, c2] add into the same accumulator: the decoder's conv over torch.cat((up, skip)) as two convs, with
// the weight split on the host, so no concat buffer is written
//
// M index r runs over padded rows: output (z, y, x) is the window whose first padded voxel is (z, y, x), row
// z plane + y (w + 2) + x, and its centre (z + 1, y + 1, x + 1) is row r + plane + (w + 2) + 1 of the padded output.
// The grid tiles each z's h (w + 2) rows, so the z-padding rows are never computed and only the 2 x-padding rows of
// each line are; grid (o tiles, d * tiles per plane), 128 threads
template <int BM, int BN>
kernel void taps_s1(device float* p_x [[buffer(0)]], device float* p_w [[buffer(1)]], device float* p_x2 [[buffer(2)]],
                    device float* p_w2 [[buffer(3)]], device float* p_y [[buffer(4)]],
                    device const float* p_bias [[buffer(5)]], constant uint& d [[buffer(6)]],
                    constant uint& h [[buffer(7)]], constant uint& w [[buffer(8)]], constant uint& c [[buffer(9)]],
                    constant uint& c2 [[buffer(10)]], constant uint& o [[buffer(11)]],
                    constant uint& xrows [[buffer(12)]], constant uint& flags [[buffer(13)]],
                    uint2 tg [[threadgroup_position_in_grid]]) {
    int wp = w + 2, plane = (h + 2) * wp;
    int span = h * wp, per_plane = (span + BM - 1) / BM;
    int z = tg.y / per_plane, chunk = tg.y % per_plane;
    int base = z * plane;
    int m0 = base + chunk * BM, n0 = tg.x * BN;
    int end = base + span;
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    // a K = 3c view starting at row r reads rows r .. r + 2, so the extent stops 2 rows short of the allocation
    mat<float> tx(p_x, dextents<int32_t, 2>(3 * c, xrows - 2), array<int, 2>{1, (int)c});
    mat<float> tw0(p_w, dextents<int32_t, 2>(3 * c, o), array<int, 2>{1, 27 * (int)c});
    auto acc = op.template get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(tw0.slice(0, 0)), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    for (int t = 0; t < 9; t++) {
        int shift = (t / 3) * plane + (t % 3) * wp;
        mat<float> tw(p_w + t * 3 * c, dextents<int32_t, 2>(3 * c, o), array<int, 2>{1, 27 * (int)c});
        auto a = tx.slice(0, m0 + shift);
        auto b = tw.slice(0, n0);
        op.run(a, b, acc);
    }
    if (c2 > 0) {
        mat<float> tx2(p_x2, dextents<int32_t, 2>(3 * c2, xrows - 2), array<int, 2>{1, (int)c2});
        for (int t = 0; t < 9; t++) {
            int shift = (t / 3) * plane + (t % 3) * wp;
            mat<float> tw(p_w2 + t * 3 * c2, dextents<int32_t, 2>(3 * c2, o), array<int, 2>{1, 27 * (int)c2});
            auto a = tx2.slice(0, m0 + shift);
            auto b = tw.slice(0, n0);
            op.run(a, b, acc);
        }
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0], r = m0 + idx[1];
        int x = (r - base) % wp;
        if (r < end && x < (int)w && col < (int)o) {
            ulong oi = (ulong)(r + plane + wp + 1) * o + col;
            p_y[oi] = conv_out(acc[i], col, p_bias, flags);
        }
    }
}

// 3x3x3 conv, pad 1, stride (sz, sy, sx) of 1 or 2 per axis: x padded [din + 2, hin + 2, win + 2, c] with `xrows`
// allocated rows, weights [o, 27, c], output y padded [dout + 2, hout + 2, wout + 2, o]. Output (z, y, x) reads the
// padded input window from (sz z, sy y, sx x), so one output line (z, y) is a row view of stride sx c over its input
// line. Grid (o tiles, dout * hout * tiles per line), 128 threads
template <int BM, int BN>
kernel void taps_sn(device float* p_x [[buffer(0)]], device float* p_w [[buffer(1)]], device float* p_y [[buffer(2)]],
                    device const float* p_bias [[buffer(3)]], constant uint& din [[buffer(4)]],
                    constant uint& hin [[buffer(5)]], constant uint& win [[buffer(6)]], constant uint& c [[buffer(7)]],
                    constant uint& dout [[buffer(8)]], constant uint& hout [[buffer(9)]],
                    constant uint& wout [[buffer(10)]], constant uint& o [[buffer(11)]], constant uint& sz [[buffer(12)]],
                    constant uint& sy [[buffer(13)]], constant uint& sx [[buffer(14)]],
                    constant uint& flags [[buffer(15)]], uint2 tg [[threadgroup_position_in_grid]]) {
    int wpi = win + 2, planei = (hin + 2) * wpi;
    int per_line = (wout + BM - 1) / BM;
    int line = tg.y / per_line, chunk = tg.y % per_line;
    int zo = line / hout, yo = line % hout;
    int x0 = chunk * BM, n0 = tg.x * BN;
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    mat<float> tw0(p_w, dextents<int32_t, 2>(3 * c, o), array<int, 2>{1, 27 * (int)c});
    mat<float> tx0(p_x, dextents<int32_t, 2>(3 * c, wout), array<int, 2>{1, (int)(sx * c)});
    auto acc = op.template get_destination_cooperative_tensor<decltype(tx0.slice(0, 0)), decltype(tw0.slice(0, 0)), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    for (int t = 0; t < 9; t++) {
        int dz = t / 3, dy = t % 3;
        ulong start = ((ulong)(sz * zo + dz) * planei + (sy * yo + dy) * wpi) * c;
        // row x of this view starts at padded voxel sx x of the input line, and its K = 3c covers sx x .. sx x + 2,
        // inside the line's w + 2 voxels for every x < wout
        mat<float> tx(p_x + start, dextents<int32_t, 2>(3 * c, wout), array<int, 2>{1, (int)(sx * c)});
        mat<float> tw(p_w + t * 3 * c, dextents<int32_t, 2>(3 * c, o), array<int, 2>{1, 27 * (int)c});
        auto a = tx.slice(0, x0);
        auto b = tw.slice(0, n0);
        op.run(a, b, acc);
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0], x = x0 + idx[1];
        if (x < (int)wout && col < (int)o) {
            ulong oi = (ulong)prow(zo, yo, x, hout, wout) * o + col;
            p_y[oi] = conv_out(acc[i], col, p_bias, flags);
        }
    }
}

// y[m, n] = x[m, k] . w[n, k]^T into plain y [m, n]: the bias-free 1x1 skip projections, on the pooled skip input.
// Grid (n tiles, m tiles), 128 threads
template <int BM, int BN>
kernel void gemm_nt(device float* p_x [[buffer(0)]], device float* p_w [[buffer(1)]], device float* p_y [[buffer(2)]],
                    constant uint& m [[buffer(3)]], constant uint& n [[buffer(4)]], constant uint& k [[buffer(5)]],
                    uint2 tg [[threadgroup_position_in_grid]]) {
    int m0 = tg.y * BM, n0 = tg.x * BN;
    mat<float> tx(p_x, dextents<int32_t, 2>(k, m), array<int, 2>{1, (int)k});
    mat<float> tw(p_w, dextents<int32_t, 2>(k, n), array<int, 2>{1, (int)k});
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mx = tx.slice(0, m0);
    auto mw = tw.slice(0, n0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mx), decltype(mw), float>();
    op.run(mx, mw, acc);
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0], row = m0 + idx[1];
        if (row < (int)m && col < (int)n) p_y[(ulong)row * n + col] = acc[i];
    }
}

// The transposed conv of kernel = stride (fz, fy, fx) as one GEMM, its scatter and bias in the epilogue: x is the
// coarser level's padded activation [din + 2, hin + 2, win + 2, k] read as a [rows, k] matrix (border rows are zero
// and computed, then dropped: the coarse levels are small), w is [taps * cout, k] (row tap * cout + co, tap
// (a fy + b) fx + c), and the output goes to the finer level's padded buffer [fz din + 2, fy hin + 2, fx win + 2,
// cout] at voxel (fz z + a, fy y + b, fx x + c), plus the bias, added on its own as PyTorch adds it. Grid (n tiles,
// rows tiles), 128 threads
template <int BM, int BN>
kernel void tconv(device float* p_x [[buffer(0)]], device float* p_w [[buffer(1)]], device float* p_y [[buffer(2)]],
                  device const float* p_bias [[buffer(3)]], constant uint& din [[buffer(4)]],
                  constant uint& hin [[buffer(5)]], constant uint& win [[buffer(6)]], constant uint& k [[buffer(7)]],
                  constant uint& cout [[buffer(8)]], constant uint& fz [[buffer(9)]], constant uint& fy [[buffer(10)]],
                  constant uint& fx [[buffer(11)]], uint2 tg [[threadgroup_position_in_grid]]) {
    uint wpi = win + 2, hpi = hin + 2;
    uint rows = (din + 2) * hpi * wpi;
    uint taps = fz * fy * fx, n = taps * cout;
    int m0 = tg.y * BM, n0 = tg.x * BN;
    mat<float> tx(p_x, dextents<int32_t, 2>(k, rows), array<int, 2>{1, (int)k});
    mat<float> tw(p_w, dextents<int32_t, 2>(k, n), array<int, 2>{1, (int)k});
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mx = tx.slice(0, m0);
    auto mw = tw.slice(0, n0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mx), decltype(mw), float>();
    op.run(mx, mw, acc);
    uint hout = fy * hin, wout = fx * win;
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        uint col = n0 + idx[0], row = m0 + idx[1];
        if (row >= rows || col >= n) continue;
        uint px = row % wpi, r = row / wpi;
        uint py = r % hpi, pz = r / hpi;
        if (px < 1 || px > win || py < 1 || py > hin || pz < 1 || pz > din) continue;
        uint tap = col / cout, co = col % cout;
        uint a = tap / (fy * fx), b = (tap / fx) % fy, cc = tap % fx;
        uint z = fz * (pz - 1) + a, y = fy * (py - 1) + b, x = fx * (px - 1) + cc;
        p_y[(ulong)prow(z, y, x, hout, wout) * cout + co] = acc[i] + p_bias[co];
    }
}

#define CONV_TILES(BM, BN)                                                                                           \
    template [[host_name("taps_s1_" #BM "x" #BN)]] kernel void taps_s1<BM, BN>(                                     \
        device float*, device float*, device float*, device float*, device float*, device const float*,            \
        constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,             \
        constant uint&, constant uint&, uint2);                                                                     \
    template [[host_name("taps_sn_" #BM "x" #BN)]] kernel void taps_sn<BM, BN>(                                     \
        device float*, device float*, device float*, device const float*, constant uint&, constant uint&,           \
        constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,             \
        constant uint&, constant uint&, constant uint&, constant uint&, uint2);                                     \
    template [[host_name("gemm_nt_" #BM "x" #BN)]] kernel void gemm_nt<BM, BN>(                                     \
        device float*, device float*, device float*, constant uint&, constant uint&, constant uint&, uint2);        \
    template [[host_name("tconv_" #BM "x" #BN)]] kernel void tconv<BM, BN>(                                         \
        device float*, device float*, device float*, device const float*, constant uint&, constant uint&,           \
        constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint2);
CONV_TILES(64, 64)
CONV_TILES(64, 32)
CONV_TILES(32, 64)
CONV_TILES(32, 32)
