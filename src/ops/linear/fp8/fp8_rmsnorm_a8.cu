// Row-resident normalization followed by BF16 rounding and quantization, adapted
// from TileKernels norm_forward_cuda.py at c1cf914a82e6901c679445f872051ce44ac2cc63.
// https://github.com/deepseek-ai/TileKernels
//
// MIT License
// Copyright (c) 2026 DeepSeek
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_a8_quantize.cuh"
#include "ops/common/warp.cuh"
#include "core/device.h"
#include "core/pdl.cuh"

#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <bool Residual, bool StoreResidual>
__global__ __launch_bounds__(256) void fp8_rmsnorm_quantize_kernel(
    const __nv_bfloat162* input, const __nv_bfloat162* norm_weight,
    const __nv_bfloat162* residual, __nv_bfloat162* residual_sum,
    std::uint8_t* codes, float* scales, float* centers, float eps) {
    constexpr int threads = 256, hidden = 5120, pairs = 10;
    const int token = blockIdx.x, tid = threadIdx.x;
    const std::int64_t base = std::int64_t(token) * (hidden / 2);
    __nv_bfloat162 values[pairs], weights[pairs];
    __shared__ float warp_sums[threads / 32];
    __shared__ float inv_shared;
    __shared__ Fp8QuantizeShared<threads> quantize_shared;
    pdl::enter();
    float sum = 0.0F;
#pragma unroll
    for (int item = 0; item < pairs; ++item) {
        const int pair = tid + item * threads;
        values[item] = input[base + pair];
        if constexpr (Residual) {
            const float2 x = __bfloat1622float2(values[item]);
            const float2 r = __bfloat1622float2(residual[base + pair]);
            values[item] = __floats2bfloat162_rn(x.x + r.x, x.y + r.y);
            if constexpr (StoreResidual) residual_sum[base + pair] = values[item];
        }
        weights[item] = norm_weight[pair];
        const float2 value = __bfloat1622float2(values[item]);
        // Preserve the existing D5120 RMSNorm's pair and CTA reduction order.
        sum += value.x * value.x + value.y * value.y;
    }
    const float block_sum = block_reduce_sum<threads>(sum, warp_sums);
    if (tid == 0) inv_shared = rsqrtf(block_sum / float(hidden) + eps);
    __syncthreads();
    const float inv = inv_shared;
    float2 normalized[pairs];
#pragma unroll
    for (int item = 0; item < pairs; ++item) {
        const float2 value = __bfloat1622float2(values[item]);
        const float2 weight = __bfloat1622float2(weights[item]);
        const float gain0 = weight.x + 1.0F, gain1 = weight.y + 1.0F;
        // This BF16 boundary is observable to the standalone FP8 quantizer.
        const __nv_bfloat162 rounded = __floats2bfloat162_rn(
            value.x * inv * gain0, value.y * inv * gain1);
        normalized[item] = __bfloat1622float2(rounded);
    }
    fp8_quantize_row<true, threads>(normalized,
        reinterpret_cast<std::uint16_t*>(codes) + base, scales, centers, token, quantize_shared);
}

} // namespace

void launch_fp8_a8_fused_rmsnorm(const Tensor& input, const Tensor& norm_weight, float eps,
                                 Fp8A8Workspace workspace, cudaStream_t stream,
                                 const Tensor* residual, Tensor* residual_sum) {
    if (!current_device_ada_fp8() || input.ne[0] != 5120 ||
        !workspace.codes || !workspace.scales || !workspace.centers)
        throw std::invalid_argument("fused RMSNorm FP8 requires Ada D5120 operand workspace");
    const auto launch = [&]<bool Residual, bool StoreResidual>() {
        CUDA_CHECK(pdl::launch_consumer({dim3(input.ne[1]), dim3(256), 0, stream},
            fp8_rmsnorm_quantize_kernel<Residual, StoreResidual>,
            static_cast<const __nv_bfloat162*>(input.data),
            static_cast<const __nv_bfloat162*>(norm_weight.data),
            residual ? static_cast<const __nv_bfloat162*>(residual->data) : nullptr,
            residual_sum ? static_cast<__nv_bfloat162*>(residual_sum->data) : nullptr,
            workspace.codes, workspace.scales, workspace.centers, eps));
    };
    if (residual_sum) launch.template operator()<true, true>();
    else if (residual) launch.template operator()<true, false>();
    else launch.template operator()<false, false>();
}

} // namespace ninfer::ops::detail
