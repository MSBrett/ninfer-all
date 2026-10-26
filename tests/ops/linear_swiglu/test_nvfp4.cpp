#include "core/weight.h"
#include "ops/common/device_tier.h"
#include "ops/common/device_route.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/common/route_table.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <array>
#include <exception>
#include <iostream>
#include <optional>
#include <string>

namespace {
// Upstream's routes keep the gate and up projections in FP32 through the activation, where most
// of this line's round them to BF16 first, so the check runs on the unified table.
struct UnifiedTable {
    UnifiedTable() {
        ninfer::ops::detail::force_linear_route_table(
            ninfer::ops::detail::LinearRouteTable::Unified);
    }

    ~UnifiedTable() { ninfer::ops::detail::force_linear_route_table(std::nullopt); }
};

int check_negative_gate() {
    using namespace ninfer;
    // A card that cannot select the unified table (unified_linear_routes) has no such route.
    if (!ops::detail::unified_linear_routes_here()) { return 0; }
    const UnifiedTable unified;
    namespace qw = test::quantized_weight;
    if (test::cuda_unavailable()) return 0;
    constexpr int rows = 34816, k = 5120, half = rows / 2, max_tokens = 257;
    auto packed = qw::make_patterned_weight(
        QType::NVFP4, rows, k, 1837U, {.weight_scale_divisor = 1.0f, .input_scale_divisor = 6.0f});
    std::fill(packed.payload.begin(), packed.payload.end(), 0);
    // Preserve the scalar metadata, while constructing the two independently scaled groups.
    const float divisor = 1.0f;
    std::memcpy(packed.payload.data() + packed.weight_divisor_offset, &divisor, 4);
    const auto scale_offset = [&](int row, int group) {
        return packed.scale_plane_offset +
               static_cast<std::size_t>((row / 128) * (k / 64) + group / 4) * 512 +
               (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
    };
    for (int row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * (k / 2);
        if (row < half) {
            packed.payload[base]                 = 0x0f; // -6
            packed.payload[base + 8]             = 0x01; // +0.5 in the next G16 group
            packed.payload[scale_offset(row, 0)] = 0x46; // 3.5
            packed.payload[scale_offset(row, 1)] = 0x18; // 0.0625
        } else {
            packed.payload[base]                 = 0x07; // +6
            packed.payload[scale_offset(row, 0)] = 0x7e; // 448
        }
    }
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * max_tokens, 0);
    for (int token = 0; token < max_tokens; ++token)
        activation[static_cast<std::size_t>(token) * k] =
            activation[static_cast<std::size_t>(token) * k + 16] = test::f32_to_bf16(1.0F);
    test::GuardedDeviceBuffer weights(packed.payload.size()), x(activation.size() * 2);
    test::GuardedDeviceBuffer y(static_cast<std::size_t>(half) * max_tokens * 2);
    weights.copy_from_host(packed.payload.data(), packed.payload.size());
    x.copy_from_host(activation.data(), activation.size() * 2);
    const Weight weight = packed.device_weight(weights.data());
    const double gate =
        qw::logical_weight_fp64(packed, 0, 0) + qw::logical_weight_fp64(packed, 0, 16);
    const double up    = qw::logical_weight_fp64(packed, half, 0);
    const double ideal = gate / (1.0 + std::exp(-gate)) * up;
    const auto bytes   = ops::linear_swiglu_workspace_capacity_bytes(
        weight.qtype, rows, k, ops::LinearPolicy::AllowA4, 1, max_tokens);
    // Zero-scratch A16 routes still need a nonempty arena object, matching
    // the shared test harness convention; the Op consumes none of its bytes.
    WorkspaceArena workspace(std::max<std::size_t>(bytes,256));
    int failures = 0;
    for (int tokens : {1, 64, 129, 256, 257}) {
        const std::string label = "NVFP4 SwiGLU negative gate T=" + std::to_string(tokens);
        failures += test::run_case_allowing_arch_skip(label, [&] {
            Tensor input(x.data(), DType::BF16, {k, tokens});
            Tensor output(y.data(), DType::BF16, {half, tokens});
            ops::linear_swiglu(input, weight, output, ops::LinearPolicy::AllowA4, workspace,
                               nullptr);
            test::cuda_synchronize();
            std::vector<std::uint16_t> bits(static_cast<std::size_t>(half) * tokens);
            y.copy_to_host(bits.data(), bits.size() * 2);
            std::vector<double> got(bits.size()), reference(bits.size(), ideal);
            for (std::size_t i = 0; i < bits.size(); ++i) got[i] = test::bf16_to_f32(bits[i]);
            // This input is exactly representable by A4. The negative gate makes loss of
            // projection precision observable in the complete nonlinear formula.
            return test::verify_reduction(label, got, reference, {1.0 / 256.0, 0.0, 2.0 / 256.0});
        });
    }
    return failures;
}
} // namespace

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        constexpr std::array<std::int32_t, 15> kA16Cases{
            1,4,8,16,17,32,33,64,65,96,97,128,129,257,1025};
        // 511 and 513 straddle the ragged floor: the first still reaches the composition, the
        // second is the narrowest ragged width the fused route admits and leaves one real token
        // in a third M tile. 767 leaves that tile all but full, and 1025 leaves one after four
        // whole tiles. 255/256 straddle the whole-tile floor, which does not move; 257 stays as
        // the width just past it that the composition keeps.
        constexpr std::array<std::int32_t, 21> kA4Cases{2,   4,   5,   16,  56,  64,   65,
                                                        96,  97,  112, 128, 129, 255,  256,
                                                        257, 511, 512, 513, 767, 1024, 1025};
        int failures = check_negative_gate();
        failures += run_profile("LinearSwiGLU NVFP4_A16",
                                {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16},
                                kA16Cases, std::array<std::int32_t,3>{17,65,1025});
        failures += run_profile("LinearSwiGLU NVFP4_A4",
                                {QType::NVFP4, 34816, 5120, 17408, 1803U, ActivationCompute::A4},
                                kA4Cases, std::array<std::int32_t, 4>{65, 97, 128, 129});
        if (ops::current_device_ada_fp8()) {
            const ops::DeviceRouteForce candidate("nvfp4_linear_swiglu/34816x5120","a8");
            constexpr std::array<std::int32_t, 12> a8_cases{1, 3, 8, 16, 17, 32, 33, 64, 65, 129, 257, 1025};
            constexpr std::array<std::int32_t, 4> a8_graphs{1, 17, 65, 1025};
            failures += run_profile("LinearSwiGLU NVFP4_A8",
                {QType::NVFP4, 34816, 5120, 17408, 1811U, ActivationCompute::A8, false, true},
                a8_cases, a8_graphs);
            failures += run_profile("LinearSwiGLU NVFP4_A8 split divisors",
                {QType::NVFP4, 34816, 5120, 17408, 1813U, ActivationCompute::A8, true, true},
                a8_cases, a8_graphs);
            failures += run_profile("LinearSwiGLU NVFP4_A16 split divisors",
                {QType::NVFP4, 34816, 5120, 17408, 1813U, ActivationCompute::A16, true, true},
                a8_cases, a8_graphs);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
