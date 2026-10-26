#include "core/weight.h"
#include "ops/common/blackwell_routes.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_a8_quantize.cuh"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class ActivationGeometry, bool Centered, int Threads = 256>
__global__ __launch_bounds__(Threads,
                             2) void fp8_a8_quantize_kernel(const __nv_bfloat16* __restrict__ input,
                                                            std::uint8_t* __restrict__ codes,
                                                            float* __restrict__ scales,
                                                            float* __restrict__ centers) {
    static_assert((ActivationGeometry::kInputRows % (Threads * 2)) == 0);
    constexpr int pairs_per_token  = ActivationGeometry::kInputRows / 2;
    constexpr int pairs_per_thread = pairs_per_token / Threads;
    __shared__ Fp8QuantizeShared<Threads> shared;
    pdl::enter();
    const int token = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const auto* input_pairs = reinterpret_cast<const std::uint32_t*>(
        input + static_cast<std::int64_t>(token) * ActivationGeometry::kInputRows);
    auto* output_pairs = reinterpret_cast<std::uint16_t*>(
        codes + static_cast<std::int64_t>(token) * ActivationGeometry::kInputRows);
    float2 values[pairs_per_thread];
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item)
        values[item] = bf16x2_bits_to_float2(input_pairs[tid + item * Threads]);
    fp8_quantize_row<Centered, Threads>(values, output_pairs, scales, centers, token, shared);
}

template <class ActivationGeometry>
void launch_quantize_exact(const Tensor& x, Fp8A8Workspace workspace, cudaStream_t stream) {
    constexpr int kThreads = 256;
    const auto launch = [&]<bool Centered>() {
        CUDA_CHECK(pdl::launch_consumer({dim3(x.ne[1]), dim3(kThreads), 0, stream},
                                        fp8_a8_quantize_kernel<ActivationGeometry, Centered, kThreads>,
                                        static_cast<const __nv_bfloat16*>(x.data), workspace.codes,
                                        workspace.scales, workspace.centers));
    };
    if (workspace.centers) launch.template operator()<true>();
    else launch.template operator()<false>();
}

} // namespace

void launch_fp8_a8_quantize(const Tensor& x, const Weight& weight, Fp8A8Workspace workspace,
                            cudaStream_t stream) {
    require_fp8_a8_card();
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("fp8 A8 requires caller workspace");
    }
    switch (weight.k) {
    case Fp8Activation5120Geometry::kInputRows:
        launch_quantize_exact<Fp8Activation5120Geometry>(x, workspace, stream);
        return;
    case Fp8Activation6144Geometry::kInputRows:
        launch_quantize_exact<Fp8Activation6144Geometry>(x, workspace, stream);
        return;
    case Fp8Activation17408Geometry::kInputRows:
        launch_quantize_exact<Fp8Activation17408Geometry>(x, workspace, stream);
        return;
    default:
        throw std::invalid_argument("fp8 A8 quantize: unsupported K");
    }
}

} // namespace ninfer::ops::detail
