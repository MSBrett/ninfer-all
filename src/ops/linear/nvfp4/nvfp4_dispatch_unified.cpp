#include "ops/linear/nvfp4/nvfp4_dispatch_unified.h"
#include "ops/linear/nvfp4/nvfp4_shapes_unified.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_a8_plan.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail::unified {
namespace {
const std::array kShapes{&kNvfp4N14336K5120, &kNvfp4N16384K5120, &kNvfp4N34816K5120,
                         &kNvfp4N5120K6144, &kNvfp4N5120K17408, &kNvfp4N248320K5120, &kNvfp4N131072K5120};

const Nvfp4LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("nvfp4 linear: unsupported policy");
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("nvfp4 linear: unsupported shape");
}
} // namespace

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(n, k, policy);
    if (allows_a8(policy) && current_device_ada_fp8())
        return nvfp4_ada_a8_capacity(policy,Nvfp4AdaOp::Linear,n,k,min_tokens,max_tokens);
    return allows_a4(policy) && shape.uses_a4(min_tokens, max_tokens)
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, k)
               : 0;
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    if (nvfp4_uses_ada_a8(policy,Nvfp4AdaOp::Linear,weight.n,weight.k,x.ne[1])) {
        validate_nvfp4_weight(weight,"nvfp4 linear");
        (void)resolve_shape(weight.n,weight.k,policy);
        if (x.ne[1] <= 0) throw std::invalid_argument("nvfp4 linear: T must be positive");
        if (!workspace) throw std::invalid_argument("nvfp4 A8 linear requires caller workspace");
        nvfp4_a8_linear_launch(x,weight,out,*workspace,stream);
        return;
    }
    policy = nvfp4_ada_fallback_policy(policy);
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("nvfp4 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    if (!allows_a4(policy) || !shape.uses_a4(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("nvfp4 A4 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    shape.a4(x, weight, out, scratch, stream);
}
} // namespace ninfer::ops::detail::unified
