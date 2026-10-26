#pragma once

#include "ops/common/warp.cuh"
#include <cuda_fp8.h>
#include <math_constants.h>

namespace ninfer::ops::detail {

template <int Threads> struct Fp8QuantizeShared {
    float maxima[Threads / 32];
    float minima[Threads / 32];
    float upper[Threads / 32];
    float scale;
    float center;
};

// Both standalone and fused producers quantize the same represented BF16 row.
template <bool Centered, int Threads, int Pairs>
__device__ __forceinline__ void fp8_quantize_row(
    const float2 (&values)[Pairs], std::uint16_t* output, float* scales, float* centers,
    int token, Fp8QuantizeShared<Threads>& shared) {
    constexpr int warps = Threads / 32;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float maximum = 0.0F, lower = CUDART_INF_F, upper = -CUDART_INF_F;
#pragma unroll
    for (int item = 0; item < Pairs; ++item) {
        maximum = fmaxf(maximum, fabsf(values[item].x));
        maximum = fmaxf(maximum, fabsf(values[item].y));
        if constexpr (Centered) {
            lower = fminf(lower, fminf(values[item].x, values[item].y));
            upper = fmaxf(upper, fmaxf(values[item].x, values[item].y));
        }
    }
    maximum = warp_max(maximum);
    if (lane == 0) shared.maxima[warp] = maximum;
    if constexpr (Centered) {
        lower = -warp_max(-lower);
        upper = warp_max(upper);
        if (lane == 0) { shared.minima[warp] = lower; shared.upper[warp] = upper; }
    }
    __syncthreads();
    if (warp == 0) {
        maximum = warp_max(lane < warps ? shared.maxima[lane] : 0.0F);
        if constexpr (Centered) {
            lower = -warp_max(lane < warps ? -shared.minima[lane] : -CUDART_INF_F);
            upper = warp_max(lane < warps ? shared.upper[lane] : -CUDART_INF_F);
            if (lane == 0) {
                shared.center = lower > 0.0F || upper < 0.0F
                                    ? __fadd_rn(0.5F * lower, 0.5F * upper) : 0.0F;
                const float residual_max = shared.center == 0.0F ? maximum
                    : fmaxf(fabsf(__fsub_rn(lower, shared.center)),
                            fabsf(__fsub_rn(upper, shared.center)));
                shared.scale = residual_max > 0.0F ? residual_max / 448.0F : 0.0F;
            }
        } else if (lane == 0) {
            shared.scale = maximum > 0.0F ? maximum / 448.0F : 0.0F;
        }
    }
    __syncthreads();
    const float scale = shared.scale;
    const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
    const float center = Centered ? shared.center : 0.0F;
#pragma unroll
    for (int item = 0; item < Pairs; ++item) {
        const float2 residual = Centered
            ? make_float2(__fsub_rn(values[item].x, center), __fsub_rn(values[item].y, center))
            : values[item];
        const float2 scaled = isfinite(inverse)
            ? make_float2(residual.x * inverse, residual.y * inverse)
            : make_float2(__fdiv_rn(residual.x, scale), __fdiv_rn(residual.y, scale));
        output[tid + item * Threads] = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
    }
    if (tid == 0) {
        scales[token] = scale;
        if constexpr (Centered) centers[token] = center;
    }
}

} // namespace ninfer::ops::detail
