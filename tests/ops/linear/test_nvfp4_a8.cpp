#include "ops/common/device_tier.h"
#include "ops/linear/nvfp4/nvfp4_a8_plan.h"
#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <utility>

namespace {
using namespace ninfer;
using namespace ninfer::test::linear;

int qualify() {
    constexpr std::array calls{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{3, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{5, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{31, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{32, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{33, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    constexpr std::array biased_calls{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    int failures = 0;
    unsigned seed = 1901;
    for (const auto [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120},
                              std::pair{34816, 5120}, std::pair{5120, 6144},
                              std::pair{5120, 17408}}) {
        const ops::DeviceRouteForce candidate(
            ops::detail::nvfp4_ada_route_key(ops::detail::Nvfp4AdaOp::Linear,n,k),"a8");
        failures += run_shape("NVFP4_A8", ActivationCompute::A8, make_nvfp4_weight,
            {n, k, seed++, Comparison::SampledRows, true, calls,
             ActivationSigns::Centered, DType::BF16, true});
        failures += run_shape("NVFP4_A8 biased", ActivationCompute::A8, make_nvfp4_weight,
            {n, k, seed++, Comparison::SampledRows, true, biased_calls,
             ActivationSigns::Biased, DType::BF16, true});
        failures += verify_workspace_envelopes(QType::NVFP4, n, k);
    }
    constexpr std::array head_a8{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{3, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{33, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    constexpr std::array head_a16{
        Invocation{1, CallForm::A16Convenience},
        Invocation{3}, Invocation{8}, Invocation{17}, Invocation{33},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::A16Only, true},
        Invocation{257},
    };
    for (const int n : {131072, 248320}) {
        const ops::DeviceRouteForce candidate(
            ops::detail::nvfp4_ada_route_key(ops::detail::Nvfp4AdaOp::Linear,n,5120),"a8");
        failures += run_shape("NVFP4 head A8", ActivationCompute::A8, make_nvfp4_weight,
            {n, 5120, seed++, Comparison::SampledRows, true, head_a8,
             ActivationSigns::Centered, DType::BF16, true});
        failures += run_shape("NVFP4 head A16", ActivationCompute::A16, make_nvfp4_weight,
            {n, 5120, seed++, Comparison::SampledRows, true, head_a16,
             ActivationSigns::Centered, DType::BF16, true});
        failures += verify_workspace_envelopes(QType::NVFP4, n, 5120);
    }
    // Auto permission must follow the production A16 choice, including the
    // AllowA4-on-Ada fallback. Keep its strict original-input A16 criterion.
    constexpr std::array auto_calls{
        Invocation{1,CallForm::Policy,ops::LinearPolicy::AllowA8},
        Invocation{17,CallForm::Policy,ops::LinearPolicy::AllowA4,true},
        Invocation{65,CallForm::Policy,ops::LinearPolicy::AllowA8},
        Invocation{1025,CallForm::Policy,ops::LinearPolicy::AllowA4,true}};
    for (const auto [n,k] : {std::pair{34816,5120},std::pair{5120,17408}})
        failures+=run_shape("NVFP4 auto A16",ActivationCompute::A16,make_nvfp4_weight,
            {n,k,1921U,Comparison::SampledRows,true,auto_calls,ActivationSigns::Biased,DType::BF16,true});

    // FP32 tensor-parallel partials must retain the A16 arithmetic even when a registered Ada
    // shape's profile selects A8 for ordinary BF16 output.
    {
        const ops::DeviceRouteForce candidate("nvfp4_linear/5120x6144", "a8");
        constexpr std::array partial_calls{
            Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
            Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
            Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        };
        failures += run_shape("NVFP4 FP32 partial A16", ActivationCompute::A16, make_nvfp4_weight,
                              {5120, 6144, 1922U, Comparison::SampledRows, true, partial_calls,
                               ActivationSigns::Centered, DType::FP32, true});
    }

    // An interval ending in A16 still has to reserve workspace for an A8
    // band inside it. Exercise the public envelope under an actual profile.
    const auto previous=ops::installed_device_route_profile();
    auto profile=std::make_shared<ops::DeviceRouteProfile>();
    profile->routes["nvfp4_linear/5120x17408"]={{32,"a16"},{65,"a8"},{1025,"a16"}};
    ops::install_device_route_profile(profile);
    failures+=verify_workspace_envelopes(QType::NVFP4,5120,17408);
    ops::install_device_route_profile(previous);
    return failures;
}
}

int main() {
    if (!cuda_available() || !ninfer::ops::current_device_ada_fp8()) {
        std::cout << "SKIP: this qualification requires loaded SM89 FP8 code\n";
        return 77;
    }
    try {
        const int failures = qualify();
        std::cout << (failures ? "FAIL" : "OK") << " NVFP4 A8 Linear\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 A8 Linear: " << error.what() << '\n';
        return 1;
    }
}
