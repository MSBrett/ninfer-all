#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan_unified.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh"
#include "ops/linear/nvfp4/nvfp4_a8_plan.h"

#include <cuda_bf16.h>

#include <array>
#include <cstddef>
#include <utility>

namespace ninfer::ops::detail::unified {
namespace {

using Geometry              = Nvfp4N34816K5120;
constexpr int kIntermediate = Geometry::kOutputRows / 2;

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SimtActivationAccess::SharedPhase
                                                  : Nvfp4SimtActivationAccess::TokenPacked;
    static constexpr int kWarpsPerCta       = ActiveTokens >= 13 ? 16 : (ActiveTokens >= 5 ? 4 : 8);
    using Schedule =
        Nvfp4A16SimtSchedule<kWarpsPerCta, 1, 2, 16, ActiveTokens, 1, kActivationAccess,
                             Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                             Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    launch_nvfp4_a16_simt<Nvfp4ScheduleInstance<Schedule, 5120, ActiveTokens, true>>(
        nvfp4_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), kIntermediate},
        Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<1>{});
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{&launch_exact<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<1>{});

} // namespace

void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    const auto p = nvfp4_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    if (x.ne[1] <= 2)
        kLaunchers[0](x, weight, out, stream);
    else if (x.ne[1] <= 8)
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 5120>>(
            p, output, Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<8>{});
    else
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 5120>>(
            p, output, Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<8>{});
}

} // namespace ninfer::ops::detail::unified

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
void launch_swiglu_matrix(const Tensor& x,const Weight& weight,Tensor& out,cudaStream_t stream) {
    using namespace unified;
    auto operands=nvfp4_a16_operands(x,weight);
    // The collective epilogue reads each stored global divisor independently.
    // Keeping the raw reduction unscaled also avoids host reciprocal overflow.
    operands.alpha=1.0F;
    using S=Nvfp4ScheduleInstance<Schedule,5120>;
    launch_nvfp4_a16_mma<S>(operands,
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data),weight.n/2},
        Nvfp4SwiGluMmaEpilogue{static_cast<const float*>(weight.weight_divisors),weight.weight_divisor_rows},
        stream,Nvfp4SwiGluRows<S::kBlockRows/2>{});
}
}

void nvfp4_a16_swiglu_full_launch(const Tensor& x,const Weight& w,Tensor& y,cudaStream_t s) {
    using namespace unified;
    // Same tensor-core mainloop and wide schedules as the optimized Linear
    // gate/up parent. Pair only CTA load rows; stored packed weights stay put.
    if (x.ne[1]<=32)
        launch_swiglu_matrix<Nvfp4A16MmaSchedule<32,32,128,16,16,1,3>>(x,w,y,s);
    else if (x.ne[1]<=64)
        launch_swiglu_matrix<Nvfp4A16MmaSchedule<32,64,128,16,16,1,3>>(x,w,y,s);
    else if (x.ne[1]<=96)
        launch_swiglu_matrix<Nvfp4A16MmaSchedule<64,96,128,64,16,1,2>>(x,w,y,s);
    else
        launch_swiglu_matrix<Nvfp4A16MmaSchedule<64,128,64,64,16,2,2>>(x,w,y,s);
}
} // namespace ninfer::ops::detail
