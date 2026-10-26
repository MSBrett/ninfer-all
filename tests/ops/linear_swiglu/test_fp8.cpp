#include "core/weight.h"
#include "core/device.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/common/route_table.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/common/device_tier.h"

#include <array>
#include <algorithm>
#include <cmath>
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
    constexpr int rows = 34816, k = 5120, half = rows / 2, max_tokens = 513;
    auto packed = qw::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, rows, k, 1817U);
    std::fill(packed.payload.begin(), packed.payload.end(), 0);
    for (int row = 0; row < rows; ++row) {
        // E4M3 -20 and +32, with independently represented BF16 row multipliers.
        packed.payload[static_cast<std::size_t>(row) * k] = row < half ? 0xda : 0x60;
        const auto scale = test::f32_to_bf16(row < half ? 1.0078125F : 1048576.0F);
        std::memcpy(packed.payload.data() + packed.scale_plane_offset + 2 * row, &scale, 2);
    }
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * max_tokens, 0);
    for (int token = 0; token < max_tokens; ++token)
        activation[static_cast<std::size_t>(token) * k] =
            test::f32_to_bf16(token % 3 == 2 ? 0.0F : 1.0F);
    test::GuardedDeviceBuffer weights(packed.payload.size()), x(activation.size() * 2);
    test::GuardedDeviceBuffer y(static_cast<std::size_t>(half) * max_tokens * 2);
    weights.copy_from_host(packed.payload.data(), packed.payload.size());
    x.copy_from_host(activation.data(), activation.size() * 2);
    const Weight weight = packed.device_weight(weights.data());
    const double gate   = qw::logical_weight_fp64(packed, 0, 0);
    const double up     = qw::logical_weight_fp64(packed, half, 0);
    const double ideal  = gate / (1.0 + std::exp(-gate)) * up;
    const auto bytes    = ops::linear_swiglu_workspace_capacity_bytes(
        weight.qtype, rows, k, ops::LinearPolicy::AllowA8, 1, max_tokens);
    WorkspaceArena workspace(bytes);
    int failures = 0;
    for (int tokens : {1, 64, 65, 257, 513}) {
        const std::string label = "FP8 SwiGLU negative gate T=" + std::to_string(tokens);
        failures += test::run_case_allowing_arch_skip(label, [&] {
            Tensor input(x.data(), DType::BF16, {k, tokens});
            Tensor output(y.data(), DType::BF16, {half, tokens});
            ops::linear_swiglu(input, weight, output, ops::LinearPolicy::AllowA8, workspace,
                               nullptr);
            test::cuda_synchronize();
            std::vector<std::uint16_t> bits(static_cast<std::size_t>(half) * tokens);
            y.copy_to_host(bits.data(), bits.size() * 2);
            std::vector<double> got(bits.size()), reference(bits.size());
            for (int token = 0; token < tokens; ++token)
                std::fill_n(reference.begin() + static_cast<std::ptrdiff_t>(token) * half,
                            half, token % 3 == 2 ? 0.0 : ideal);
            for (std::size_t i = 0; i < bits.size(); ++i) got[i] = test::bf16_to_f32(bits[i]);
            // This input is exactly representable by A8. The negative gate makes loss of
            // projection precision observable in the complete nonlinear formula.
            return test::verify_reduction(label, got, reference, {1.0 / 256.0, 0.0, 2.0 / 256.0});
        });
    }
    return failures;
}

int check_affine_center_replay() {
    using namespace ninfer;
    namespace qw=test::quantized_weight;
    if (!ops::current_device_ada_fp8()) return 0;
    constexpr int rows=34816,k=5120,half=rows/2;
    constexpr auto policy=ops::LinearPolicy::AllowA8;
    constexpr std::array<int,8> columns{0,1,15,16,k/2,k-3,k-2,k-1};
    constexpr std::array<std::uint8_t,8> codes{0x38,0xb0,0x40,0xb8,0x28,0x34,0x30,0x44};
    auto packed=qw::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16,rows,k,1841U);
    std::fill(packed.payload.begin(),packed.payload.end(),0);
    for (int row=0;row<rows;++row) {
        // Every raw code sum is +/-6. Gate/up signs and represented row
        // multipliers differ, so their affine corrections cannot be shared.
        for (int i=0;i<8;++i)
            packed.payload[static_cast<std::size_t>(row)*k+columns[i]]=
                codes[(i+row)%8] ^ (row%(row<half ? 3 : 5)==0 ? 0x80U : 0U);
        qw::detail::store_u16_le(packed.payload,
            packed.scale_plane_offset+static_cast<std::size_t>(row)*2,
            test::f32_to_bf16((1.0F+static_cast<float>(row%7)/8.0F)/(row<half ? 16.0F : 32.0F)));
    }
    test::GuardedDeviceBuffer weights(packed.payload.size());
    weights.copy_from_host(packed.payload.data(),packed.payload.size());
    const Weight weight=packed.device_weight(weights.data());
    cudaStream_t stream=nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    int failures=0;
    for (auto table : {ops::detail::LinearRouteTable::Legacy,ops::detail::LinearRouteTable::Unified}) {
        ops::detail::force_linear_route_table(table);
        for (int tokens : {33,65,129,1025}) {
            const std::size_t words=static_cast<std::size_t>(half)*tokens;
            test::GuardedDeviceBuffer dx(static_cast<std::size_t>(k)*tokens*2),dy(words*2);
            Tensor x(dx.data(),DType::BF16,{k,tokens}),y(dy.data(),DType::BF16,{half,tokens});
            const auto capacity=ops::linear_swiglu_workspace_capacity_bytes(weight.qtype,rows,k,policy,tokens,tokens);
            if (capacity==0) throw std::runtime_error("FP8 SwiGLU affine fixture did not select A8 operands");
            test::GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity,256));
            WorkspaceArena workspace(DeviceSpan{scratch.data(),scratch.bytes()});
            const auto invoke=[&] { ops::linear_swiglu(x,weight,y,policy,workspace,stream); };
            CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            invoke();
            cudaGraph_t graph=nullptr;
            cudaGraphExec_t executable=nullptr;
            CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
            CUDA_CHECK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
            for (int revision=0;revision<3;++revision) {
                std::vector<std::uint16_t> input(static_cast<std::size_t>(k)*tokens);
                for (int token=0;token<tokens;++token) {
                    const int kind=(token+revision)%6;
                    for (int col=0;col<k;++col) {
                        const float offset=static_cast<float>((col+3*token+revision)%5-2)*0.25F;
                        const float center=2.0F+0.25F*revision;
                        const float value=kind==0 ? 0.0F : kind==1 ? 0.5F*(revision+1) :
                            kind==2 ? -0.5F*(revision+1) : kind==3 ? center+offset :
                            kind==4 ? -center+offset : offset;
                        input[static_cast<std::size_t>(token)*k+col]=test::f32_to_bf16(value);
                    }
                }
                dx.copy_from_host(input.data(),input.size()*2);
                CUDA_CHECK(cudaGraphLaunch(executable,stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::vector<std::uint16_t> captured(words),eager(words);
                dy.copy_to_host(captured.data(),words*2);
                invoke();
                CUDA_CHECK(cudaStreamSynchronize(stream));
                dy.copy_to_host(eager.data(),words*2);
                const std::string label="FP8 SwiGLU affine centers T="+std::to_string(tokens)+
                    " replay="+std::to_string(revision)+
                    (table==ops::detail::LinearRouteTable::Legacy ? " legacy" : " unified");
                if (captured!=eager) { std::cerr<<label<<": eager/graph mismatch\n";++failures; }
                std::vector<double> actual,expected;
                for (int sample=0;sample<32;++sample) {
                    const int row=(half-1)*sample/31;
                    for (int token=0;token<tokens;++token) {
                        double gate=0,up=0;
                        // All omitted weights are stored zero. Decode each
                        // nonzero weight independently; no quantized inputs
                        // or private MMA reduction appear in this oracle.
                        for (int col : columns) {
                            const double value=test::bf16_to_f32(input[static_cast<std::size_t>(token)*k+col]);
                            gate+=qw::logical_weight_fp64(packed,row,col)*value;
                            up+=qw::logical_weight_fp64(packed,row+half,col)*value;
                        }
                        const double e=std::exp(-std::abs(gate));
                        const double activated=gate>=0 ? gate/(1.0+e) : gate*e/(1.0+e);
                        actual.push_back(test::bf16_to_f32(captured[static_cast<std::size_t>(token)*half+row]));
                        expected.push_back(activated*up);
                    }
                }
                failures+=test::verify_reduction(label,actual,expected,{8.0e-2,1.0e-2,1.2e-1});
                failures+=dx.verify_guards(label+" input");
                failures+=dy.verify_guards(label+" output");
                failures+=scratch.verify_guards(label+" workspace");
                if (workspace.used()!=0 || workspace.peak_used()>capacity) {
                    std::cerr<<label<<": workspace lifetime/capacity mismatch\n";++failures;
                }
            }
            CUDA_CHECK(cudaGraphExecDestroy(executable));
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
        ops::detail::force_linear_route_table(std::nullopt);
    }
    failures+=weights.verify_guards("FP8 SwiGLU affine weights");
    std::vector<std::uint8_t> after(packed.payload.size());
    weights.copy_to_host(after.data(),after.size());
    if (after!=packed.payload) { std::cerr<<"FP8 SwiGLU affine weights modified\n";++failures; }
    CUDA_CHECK(cudaStreamDestroy(stream));
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
        constexpr std::array kA16Cases{1,  2,  4,  5,  8,  9,  16,  17,  24,  25,
                                       32, 33, 64, 65, 96, 97, 128, 129, 1024};
        constexpr std::array kA8Cases{1,   2,   3,   4,   5,   6,   8,   15,  16,   17,   31,
                                      32,  33,  48,  63,  64,  65,  96,  127, 128,  129,  191,
                                      192, 193, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025};
        int failures = 0;
        failures += check_negative_gate();
        failures += check_affine_center_replay();
        failures += run_profile(
            "LinearSwiGLU FP8_A16",
            {QType::FP8_E4M3FN_ROW_BF16, 34816, 5120, 17408, 1811U, ActivationCompute::A16},
            kA16Cases, std::array<std::int32_t, 1>{16});
        failures += run_profile(
            "LinearSwiGLU FP8_A8",
            {QType::FP8_E4M3FN_ROW_BF16, 34816, 5120, 17408, 1813U, ActivationCompute::A8},
            kA8Cases, std::array{4, 5, 65, 193, 257, 512, 513, 1025});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU FP8 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU FP8 test failed: " << error.what() << '\n';
        return 1;
    }
}
