#pragma once
#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/device_tier.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/common/device_route.h"

#include <algorithm>

namespace ninfer::ops::detail {
struct Nvfp4A8Workspace {
    std::uint8_t* codes;
    float* scales;
    float* centers;
};

// Only activations are materialized. The extra center plane is FP32 per token;
// resident packed weight codes/scales and global divisors never expand.
template<class Arena>
Nvfp4A8Workspace allocate_nvfp4_a8_workspace(Arena& arena,int tokens,int input_rows) {
    if (input_rows <= 0 || input_rows % 32)
        throw std::invalid_argument("nvfp4 A8 workspace: invalid K");
    const auto codes=arena.alloc_bytes(fp8_a8_checked_bytes(tokens,input_rows),256);
    const auto scales=arena.alloc_bytes(fp8_a8_checked_bytes(tokens,sizeof(float)),256);
    const auto centers=arena.alloc_bytes(fp8_a8_checked_bytes(tokens,sizeof(float)),256);
    return {static_cast<std::uint8_t*>(codes.data),static_cast<float*>(scales.data),
            static_cast<float*>(centers.data)};
}
inline std::size_t nvfp4_a8_workspace_capacity_bytes(int tokens,int input_rows) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_nvfp4_a8_workspace(layout,tokens,input_rows);
    return layout.peak_bytes(1);
}

enum class Nvfp4AdaOp { Linear, SwiGlu, Add };

inline const char* nvfp4_ada_route_key(Nvfp4AdaOp op, int n, int k) {
    if (op == Nvfp4AdaOp::SwiGlu)
        return n==34816 && k==5120 ? "nvfp4_linear_swiglu/34816x5120" : "";
    if (op == Nvfp4AdaOp::Add) {
        if (n==5120 && k==6144) return "nvfp4_linear_add/5120x6144";
        if (n==5120 && k==17408) return "nvfp4_linear_add/5120x17408";
        return "";
    }
    if (k==5120) {
        switch (n) {
        case 14336: return "nvfp4_linear/14336x5120";
        case 16384: return "nvfp4_linear/16384x5120";
        case 34816: return "nvfp4_linear/34816x5120";
        case 131072: return "nvfp4_linear/131072x5120";
        case 248320: return "nvfp4_linear/248320x5120";
        }
    }
    if (n==5120 && k==6144) return "nvfp4_linear/5120x6144";
    if (n==5120 && k==17408) return "nvfp4_linear/5120x17408";
    return "";
}

// RTX 4090 cold-cache measurements select A16 by default: the packed G16 A8
// candidate lost 19/20 Linear comparisons, including every head and wide call.
// Explicit device profiles / DeviceRouteForce retain direct A8 qualification
// and measurement. Permission to use A8 does not require choosing it.
inline bool nvfp4_uses_ada_a8(LinearPolicy policy, Nvfp4AdaOp op, int n, int k, int tokens) {
    return allows_a8(policy) && current_device_ada_fp8() &&
           device_route_schedule(nvfp4_ada_route_key(op,n,k),tokens) == "a8";
}

inline LinearPolicy nvfp4_ada_fallback_policy(LinearPolicy policy) {
    return current_device_ada_fp8() && allows_a8(policy) ? LinearPolicy::A16Only : policy;
}

inline std::size_t nvfp4_ada_a8_capacity(LinearPolicy policy, Nvfp4AdaOp op, int n, int k,
                                       int first, int last) {
    if (!allows_a8(policy) || !current_device_ada_fp8()) return 0;
    if (nvfp4_uses_ada_a8(policy,op,n,k,last))
        return nvfp4_a8_workspace_capacity_bytes(last,k);
    const auto profile = installed_device_route_profile();
    if (!profile) return 0;
    const auto found = profile->routes.find(nvfp4_ada_route_key(op,n,k));
    if (found == profile->routes.end()) return 0;
    int begin = 1, widest = 0;
    for (const auto& band : found->second) {
        const int end = std::min(last,band.last);
        if (end >= std::max(first,begin) && nvfp4_uses_ada_a8(policy,op,n,k,end)) widest=end;
        if (band.last >= last) break;
        begin=band.last+1;
    }
    return widest ? nvfp4_a8_workspace_capacity_bytes(widest,k) : 0;
}
void nvfp4_a8_linear_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&, cudaStream_t);
void nvfp4_a8_swiglu_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&, cudaStream_t);
void nvfp4_a8_add_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&, cudaStream_t);
// Exact public BF16 activations, for arbitrary-width A16 SwiGLU and stacked global divisors.
void nvfp4_a16_swiglu_full_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
}
