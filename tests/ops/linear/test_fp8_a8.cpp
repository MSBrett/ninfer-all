#include "core/weight.h"
#include "core/device.h"
#include "ops/common/device_route.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <optional>
#include <vector>

// A8 workspace follows the selected architecture: Ada uses shared-memory MMA schedules,
// while Blackwell may also reserve TMA split-K partials.
#if defined(NINFER_SM8X_COMPAT) && !defined(NINFER_SM120_FP8) && !defined(NINFER_SM89_FP8)
constexpr bool kA8Executable = false;
#else
constexpr bool kA8Executable = true;
#endif

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int check_ada_crossover() {
    if (!ops::current_device_ada_fp8()) return 0;
    namespace detail = ops::detail;
    struct RestoreTable {
        ~RestoreTable() { detail::force_linear_route_table(std::nullopt); }
    } restore;
    const auto capacity = [](int first, int last) {
        return ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16, 5120, 6144,
                                                    ops::LinearPolicy::AllowA8, first, last);
    };
    constexpr std::array a16_points{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{18, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{32, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    constexpr std::array a8_points{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{18, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{32, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{33, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    constexpr std::array production_a8{
        Invocation{33, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    constexpr std::array a16_permission{
        Invocation{33, CallForm::Policy, ops::LinearPolicy::A16Only, true},
    };
    int failures = 0;
    for (const auto table : {detail::LinearRouteTable::Legacy,
                             detail::LinearRouteTable::Unified}) {
        detail::force_linear_route_table(table);
        if (capacity(16, 32) != 0 || capacity(33, 33) == 0 ||
            capacity(17, 33) != capacity(33, 33)) {
            std::cerr << "Ada FP8 5120x6144 crossover workspace mismatch\n";
            ++failures;
        }
        failures += run_shape("FP8 Ada production A16", ActivationCompute::A16, make_fp8_weight,
                              {5120, 6144, 2077U, Comparison::SampledRows, true, a16_points,
                               ActivationSigns::Centered, true});
        failures += run_shape("FP8 Ada production A8", ActivationCompute::A8, make_fp8_weight,
                              {5120, 6144, 2077U, Comparison::SampledRows, true, production_a8,
                               ActivationSigns::Centered, true});
        {
            ops::DeviceRouteForce candidate("fp8_linear/5120x6144", "a8");
            if (capacity(17, 17) == 0 || capacity(16, 33) != capacity(33, 33)) {
                std::cerr << "Ada FP8 direct A8 candidate lost operand workspace\n";
                ++failures;
            }
            for (const auto signs : {ActivationSigns::Centered, ActivationSigns::Biased})
                failures += run_shape("FP8 Ada direct A8", ActivationCompute::A8, make_fp8_weight,
                                      {5120, 6144, 2077U, Comparison::SampledRows, true, a8_points,
                                       signs, true});
            if (ops::linear_workspace_capacity_bytes(
                    QType::FP8_E4M3FN_ROW_BF16, 5120, 6144,
                    ops::LinearPolicy::A16Only, 17, 33) != 0) {
                std::cerr << "Ada FP8 candidate force overrode A16Only workspace\n";
                ++failures;
            }
            failures += run_shape("FP8 Ada forced A8 respects A16Only", ActivationCompute::A16,
                                  make_fp8_weight,
                                  {5120, 6144, 2077U, Comparison::SampledRows, true, a16_permission,
                                   ActivationSigns::Centered, true});
        }
        {
            ops::DeviceRouteForce candidate("fp8_linear/5120x6144", "a16");
            if (capacity(17, 33) != 0) {
                std::cerr << "Ada FP8 direct A16 candidate reserved A8 workspace\n";
                ++failures;
            }
        }
    }
    return failures;
}

// A sparse signed identity selects exact represented BF16 inputs. Each token is compared
// separately so a large finite token cannot hide a zero-row or subnormal-row failure.
int check_directed_a8() {
    if (!ninfer::test::a8_execution_available()) return 0;
    namespace test = ninfer::test;
    constexpr int rows = 14336, k = 5120, max_tokens = 129;
    auto packed = make_fp8_weight(rows, k, 887U);
    std::fill(packed.payload.begin(), packed.payload.end(), 0);
    const auto unit = test::f32_to_bf16(1.0F);
    for (int row = 0; row < rows; ++row) {
        packed.payload[static_cast<std::size_t>(row) * k + row % 4] =
            row % 2 ? 0xb8 : 0x38; // exact E4M3 -1 or +1
        std::memcpy(packed.payload.data() + packed.scale_plane_offset + row * 2, &unit, 2);
    }
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * max_tokens, 0);
    for (int token = 0; token < max_tokens; ++token) {
        auto* x = activation.data() + static_cast<std::size_t>(token) * k;
        switch (token % 6) {
        case 0: break; // all-zero row, including the last token of both predicated calls
        case 1:
            // Largest finite BF16 and fractions.
            x[0] = 0x7f7f;
            x[1] = 0xfeff;
            x[2] = 0x7e7f;
            break;
        case 2:
            x[0] = 0x0001;
            x[1] = 0x8002;
            x[2] = 0x0004; // BF16 subnormals
            break;
        case 3:
            x[0] = test::f32_to_bf16(1.0F);
            x[1] = test::f32_to_bf16(-0.5F);
            x[2] = test::f32_to_bf16(0.25F);
            break;
        default:
            // Dense one-sided rows exercise a nonzero center and the signed
            // identity's nonzero weight-code sums, including constant rows.
            for (int col = 0; col < k; ++col)
                x[col] = test::f32_to_bf16(token % 6 == 4
                    ? 0.25F : -(0.25F + float(col % 4) * 0.125F));
            break;
        }
    }
    test::GuardedDeviceBuffer w(packed.payload.size()), x(activation.size() * 2);
    w.copy_from_host(packed.payload.data(), packed.payload.size());
    const Weight weight = packed.device_weight(w.data());
    struct RestoreTable {
        ~RestoreTable() { ops::detail::force_linear_route_table(std::nullopt); }
    } restore;
    int failures = 0;
    for (auto table : {ops::detail::LinearRouteTable::Legacy,
                       ops::detail::LinearRouteTable::Unified}) {
        ops::detail::force_linear_route_table(table);
        for (int tokens : {64, 65, 129}) {
            const auto bytes = ops::linear_workspace_capacity_bytes(
                weight.qtype, rows, k, ops::LinearPolicy::AllowA8, tokens, tokens);
            // A supported A8 route must really materialize its FP8 operands.
            if (bytes == 0) {
                std::cerr << "directed FP8 A8 reported no operand workspace\n";
                ++failures;
                continue;
            }
            WorkspaceArena workspace(bytes);
            test::GuardedDeviceBuffer y(static_cast<std::size_t>(rows) * tokens * 2);
            Tensor input(x.data(), DType::BF16, {k, tokens});
            Tensor output(y.data(), DType::BF16, {rows, tokens});
            x.copy_from_host(activation.data(), activation.size() * 2);
            ops::linear(input, weight, output, ops::LinearPolicy::AllowA8, workspace, nullptr);
            test::cuda_synchronize();
            cudaStream_t stream;
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            ops::linear(input, weight, output, ops::LinearPolicy::AllowA8, workspace, stream);
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int replay = 0; replay < 2; ++replay) {
                auto changed = activation;
                if (replay) for (auto& bits : changed) bits ^= 0x8000;
                x.copy_from_host(changed.data(), changed.size() * 2);
                CUDA_CHECK(cudaMemsetAsync(y.data(), 0x7f, y.bytes(), stream));
                CUDA_CHECK(cudaGraphLaunch(executable, stream));
                test::cuda_synchronize(stream);
                std::vector<std::uint16_t> bits(static_cast<std::size_t>(rows) * tokens);
                y.copy_to_host(bits.data(), bits.size() * 2);
                for (int token = 0; token < tokens; ++token) {
                    std::vector<double> actual(rows), reference(rows);
                    for (int row = 0; row < rows; ++row) {
                        actual[row] = test::bf16_to_f32(bits[static_cast<std::size_t>(token) * rows + row]);
                        const double operand = test::bf16_to_f32(
                            changed[static_cast<std::size_t>(token) * k + row % 4]);
                        reference[row] = row % 2 ? -operand : operand;
                    }
                    const std::string label = "directed FP8 A8 T=" + std::to_string(tokens) +
                                               " token=" + std::to_string(token);
                    failures += test::verify_reduction(
                        label, actual, reference, {1.0 / 256.0, 0.0, 2.0 / 256.0});
                }
                if (workspace.used() != 0 || workspace.peak_used() != bytes) {
                    std::cerr << "directed FP8 A8 workspace high-water mismatch\n";
                    ++failures;
                }
            }
            CUDA_CHECK(cudaGraphExecDestroy(executable));
            CUDA_CHECK(cudaGraphDestroy(graph));
            CUDA_CHECK(cudaStreamDestroy(stream));
            failures += y.verify_guards("directed FP8 A8 output");
        }
    }
    failures += w.verify_guards("directed FP8 A8 weight");
    failures += x.verify_guards("directed FP8 A8 activation");
    return failures;
}

int run_fp8_a8() {
    constexpr std::array attn_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    int failures =
        run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                  {14336, 5120, 827U, Comparison::SampledRows, true, attn_decode_invocations});
    constexpr std::array attn_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1023, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {14336, 5120, 829U, Comparison::SampledRows, true, attn_invocations});
    constexpr std::array attn_bulk_invocations{
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{288, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{289, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {14336, 5120, 833U, Comparison::SampledRows, true, attn_bulk_invocations});
    constexpr std::array gdn_decode_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {16384, 5120, 837U, Comparison::SampledRows, true, gdn_decode_invocations});
    constexpr std::array gdn_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {16384, 5120, 839U, Comparison::SampledRows, true, gdn_invocations});
    constexpr std::array gdn_bulk_invocations{
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{384, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {16384, 5120, 841U, Comparison::SampledRows, true, gdn_bulk_invocations});
    constexpr std::array mlp_decode_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{3, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {34816, 5120, 851U, Comparison::SampledRows, true, mlp_decode_invocations});
    constexpr std::array mlp_invocations{
        Invocation{5, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {34816, 5120, 853U, Comparison::SampledRows, true, mlp_invocations});

    constexpr std::array mlp_bulk_invocations{
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{511, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{2048, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {34816, 5120, 863U, Comparison::SampledRows, true, mlp_bulk_invocations});

    constexpr std::array residual6144_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape(
        "FP8_A16", ActivationCompute::A16, make_fp8_weight,
        {5120, 6144, 855U, Comparison::SampledRows, true, residual6144_decode_invocations});
    constexpr std::array residual6144_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{768, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{769, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {5120, 6144, 857U, Comparison::SampledRows, true, residual6144_invocations});
    constexpr std::array residual17408_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape(
        "FP8_A16", ActivationCompute::A16, make_fp8_weight,
        {5120, 17408, 867U, Comparison::SampledRows, true, residual17408_decode_invocations});
    constexpr std::array residual17408_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{384, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{768, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{769, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape(
        "FP8_A8", ActivationCompute::A8, make_fp8_weight,
        {5120, 17408, 859U, Comparison::SampledRows, true, residual17408_invocations});

    struct Problem {
        std::int32_t rows;
        std::int32_t input_rows;
        bool a8_at_one;
        bool a8_at_two;
    };

    for (const Problem problem :
         {Problem{14336, 5120, false, false}, Problem{16384, 5120, false, false},
          Problem{34816, 5120, false, false}, Problem{5120, 6144, false, false},
          Problem{5120, 17408, false, false}}) {
        const auto capacity = [&](std::int32_t first, std::int32_t last,
                                  ops::LinearPolicy policy = ops::LinearPolicy::AllowA8) {
            return ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16, problem.rows,
                                                        problem.input_rows, policy, first, last);
        };
        // An interval reserves what its widest-reserving width uses. A device profile may give
        // neighbouring widths different route tables, and then a narrower width can need more
        // than the widest one (a split-K schedule's partials on one table, none on the other).
        const auto widest = [&](std::int32_t first, std::int32_t last) {
            std::size_t most = 0;
            for (std::int32_t width = first; width <= last; ++width) {
                most = std::max(most, capacity(width, width));
            }
            return most;
        };
        const auto table = [](std::int32_t width) {
            return ops::detail::linear_route_table(ops::detail::LinearRouteFamily::Fp8, width);
        };
        const std::size_t one         = capacity(1, 1);
        const std::size_t two         = capacity(2, 2);
        const std::size_t forty_eight = capacity(48, 48);
        const std::size_t exact_1024  = capacity(1024, 1024);
        const std::size_t exact_1048  = capacity(1048, 1048);
        // On one table a wider A8 launch reserves more; across tables only their own sizes count.
        const bool wider_reserves_more =
            table(1024) != table(1048) || exact_1048 > exact_1024;
        if (kA8Executable &&
            ((one != 0) != problem.a8_at_one || (two != 0) != problem.a8_at_two ||
             capacity(2, 4) != 0 || forty_eight <= two || capacity(1, 48) != widest(1, 48) ||
             exact_1024 <= forty_eight || !wider_reserves_more ||
             capacity(1000, 1048) != widest(1000, 1048) ||
             capacity(1, 2048, ops::LinearPolicy::A16Only) != 0)) {
            std::cerr << "FP8 A8 workspace interval contract mismatch for N=" << problem.rows
                      << " K=" << problem.input_rows << ": 1=" << one << " 2=" << two
                      << " 2..4=" << capacity(2, 4) << " 48=" << forty_eight
                      << " 1..48=" << capacity(1, 48) << " (widest " << widest(1, 48)
                      << ") 1024=" << exact_1024 << " 1048=" << exact_1048
                      << " 1000..1048=" << capacity(1000, 1048) << " (widest "
                      << widest(1000, 1048) << ")\n";
            ++failures;
        }
    }
    failures += check_directed_a8();
    failures += check_ada_crossover();
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_fp8_a8();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 A8 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8 A8 Linear: " << error.what() << '\n';
        return 1;
    }
}
