// Native Q8 MTP FFN qualification: independently decoded FP64 mathematics, exact
// planned workspace, and graph replay at the real 27B dimensions.
#include "models/qwen3_5/execution/ffn.h"
#include "core/decode_graph.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
namespace ex = models::qwen3_5::execution;
namespace qw = test::quantized_weight;
constexpr int hidden = 5120;
constexpr int intermediate = 17408;
constexpr int nonzeros = 16;

int input_row(int token, int index) { return (token * 37 + index * 307) % hidden; }

int qualify(const qw::PackedWeight& gate_up, const qw::PackedWeight& down,
            const ex::FfnParameters& parameters, int tokens, bool replay, cudaStream_t stream) {
    std::vector<std::uint16_t> input(static_cast<std::size_t>(hidden) * tokens, 0);
    std::vector<std::uint16_t> residual(input.size());
    for (int t = 0; t < tokens; ++t) {
        for (int j = 0; j < nonzeros; ++j) {
            input[static_cast<std::size_t>(t) * hidden + input_row(t, j)] =
                test::f32_to_bf16(float((j * 7 + t * 3) % 31 - 15) * 0.002F);
        }
        for (int r = 0; r < hidden; ++r) {
            residual[static_cast<std::size_t>(t) * hidden + r] =
                test::f32_to_bf16(float((r * 3 + t) % 41 - 20) * 0.003F);
        }
    }

    test::GuardedDeviceBuffer dx(input.size() * 2), dr(residual.size() * 2);
    dx.copy_from_host(input.data(), dx.bytes());
    Tensor x(dx.data(), DType::BF16, {hidden, tokens});
    Tensor y(dr.data(), DType::BF16, {hidden, tokens});
    const auto capacity = ex::ffn_workspace_bytes(parameters, 1, tokens, true);
    test::GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    WorkspaceArena workspace(DeviceSpan{scratch.data(), capacity});
    const auto launch = [&] { ex::ffn(x, parameters, y, {}, workspace, stream, true); };

    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (replay) {
        definition.capture(stream, launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    for (int pass = 0; pass < (replay ? 2 : 1); ++pass) {
        // Replays must consume the current residual rather than a captured value.
        if (pass) {
            for (auto& value : residual) {
                value = test::f32_to_bf16(-test::bf16_to_f32(value));
            }
        }
        dr.copy_from_host(residual.data(), dr.bytes());
        if (replay) graph.launch(stream); else launch();
        test::cuda_check(cudaStreamSynchronize(stream), "MTP FFN completion");
        std::vector<std::uint16_t> output(residual.size());
        dr.copy_to_host(output.data(), dr.bytes());
        for (const auto value : output) {
            if (!std::isfinite(test::bf16_to_f32(value))) {
                throw std::runtime_error("MTP FFN produced a non-finite value");
            }
        }

        // Sparse represented input makes the complete gate/up oracle inexpensive.
        // Every intermediate channel participates in the sampled residual products.
        // First/middle/last columns cover full and partial token tiles.
        std::vector<int> columns{0, tokens / 2, tokens - 1};
        std::sort(columns.begin(), columns.end());
        columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
        std::vector<double> actual, expected;
        for (const int t : columns) {
            std::vector<double> activation(intermediate);
            for (int r = 0; r < intermediate; ++r) {
                double gate = 0.0, up = 0.0;
                for (int j = 0; j < nonzeros; ++j) {
                    const int k = input_row(t, j);
                    const double value =
                        test::bf16_to_f32(input[static_cast<std::size_t>(t) * hidden + k]);
                    gate += qw::logical_weight_fp64(gate_up, r, k) * value;
                    up += qw::logical_weight_fp64(gate_up, r + intermediate, k) * value;
                }
                const double silu = gate >= 0.0 ? gate / (1.0 + std::exp(-gate))
                                               : gate * std::exp(gate) / (1.0 + std::exp(gate));
                // The activation passed between the two Ops is an observable BF16 tensor.
                activation[r] = test::bf16_to_f32(test::f32_to_bf16(float(silu * up)));
            }
            for (int sample = 0; sample < 64; ++sample) {
                const int r = sample * (hidden - 1) / 63;
                const auto index = static_cast<std::size_t>(t) * hidden + r;
                double value = test::bf16_to_f32(residual[index]);
                for (int k = 0; k < intermediate; ++k) {
                    value += qw::logical_weight_fp64(down, r, k) * activation[k];
                }
                expected.push_back(value);
                actual.push_back(test::bf16_to_f32(output[index]));
            }
        }
        const std::string label = "Q8 MTP FFN T=" + std::to_string(tokens) +
                                  (replay ? " graph" : " eager") +
                                  " pass=" + std::to_string(pass);
        // Composition of two BF16-output contractions and a nonlinear activation:
        // 1% relative L2; gross absolute bound 0.01 and 5% of the maximum reference.
        failures += test::verify_reduction(label, actual, expected, {1.0e-2, 1.0e-2, 5.0e-2});
        failures += dr.verify_guards(label);
        failures += scratch.verify_guards(label + " workspace");
        if (workspace.used() != 0 || workspace.peak_used() > capacity) {
            throw std::runtime_error("MTP FFN exceeded or retained planned workspace");
        }
        std::cout << label << " workspace=" << capacity
                  << " peak=" << workspace.peak_used() << '\n';
    }
    failures += dx.verify_guards("MTP FFN input");
    return failures;
}
} // namespace

int main() {
    try {
        int devices = 0;
        const auto available = cudaGetDeviceCount(&devices);
        if (test::cuda_unavailable(available) || (available == cudaSuccess && devices == 0)) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        test::cuda_check(available, "cudaGetDeviceCount");
        auto gate_up = qw::make_patterned_weight(QType::Q8_G32_FP16, 2 * intermediate, hidden, 1603U);
        auto down = qw::make_patterned_weight(QType::Q8_G32_FP16, hidden, intermediate, 425U);
        test::GuardedDeviceBuffer dgu(gate_up.payload.size()), dd(down.payload.size());
        dgu.copy_from_host(gate_up.payload.data(), dgu.bytes());
        dd.copy_from_host(down.payload.data(), dd.bytes());
        ex::DenseParameters dense;
        dense.gate_up.weight = gate_up.device_weight(dgu.data());
        dense.down.weight = down.device_weight(dd.data());
        ex::FfnParameters parameters = dense;
        cudaStream_t stream = nullptr;
        test::cuda_check(cudaStreamCreate(&stream), "create MTP FFN stream");
        int failures = 0;
        for (const int tokens : {1, 8, 33, 1024}) {
            failures += qualify(gate_up, down, parameters, tokens, false, stream);
        }
        for (const int tokens : {1, 8}) {
            failures += qualify(gate_up, down, parameters, tokens, true, stream);
        }
        test::cuda_check(cudaStreamDestroy(stream), "destroy MTP FFN stream");
        failures += dgu.verify_guards("MTP FFN gate/up weights");
        failures += dd.verify_guards("MTP FFN down weights");
        std::cout << (failures ? "FAIL" : "OK") << " Q8 MTP FFN\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "MTP FFN: " << error.what() << '\n';
        return 1;
    }
}
