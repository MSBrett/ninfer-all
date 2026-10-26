#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_shapes_unified.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/common/device_route.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
const std::array kShapes{&kFp8N14336K5120, &kFp8N16384K5120, &kFp8N34816K5120,
                         &kFp8N5120K6144,  &kFp8N5120K17408, &kFp8N248320K5120, &kFp8N131072K5120};
const std::array kUnifiedShapes{&unified::kFp8N14336K5120, &unified::kFp8N16384K5120,
                                &unified::kFp8N34816K5120, &unified::kFp8N5120K6144,
                                &unified::kFp8N5120K17408, &unified::kFp8N248320K5120, &unified::kFp8N131072K5120};

const Fp8LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                    LinearRouteTable table) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
    for (const auto* shape : table == LinearRouteTable::Unified ? kUnifiedShapes : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("fp8 linear: unsupported shape");
}

bool ada_crossover_shape(const Fp8LinearShape& shape) {
    return shape.n == 5120 && shape.k == 6144 && current_device_ada_fp8();
}

bool ada_uses_a8(const Fp8LinearShape& shape, std::int32_t width) {
    // RTX 4090 cold-cache public Op, two reversed-order repetitions: A8 loses
    // at T=18/32 by 19.2/13.0%, then wins at T=33/64 by 21.8/21.3%.
    // Profiles and qualification can still select either existing candidate.
    const auto schedule = device_route_schedule("fp8_linear/5120x6144", width);
    const bool selected = schedule == "a8" || (schedule != "a16" && width >= 33);
    return selected && shape.uses_a8(width, width);
}
} // namespace

bool fp8_registered(std::int32_t n, std::int32_t k) {
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return true;
    return false;
}

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                                std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    if (!fp8_registered(n, k)) {
        if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
        return 0;
    }
    return capacity_by_table(
        min_tokens, max_tokens,
        [&](std::int32_t width) {
            return linear_route_table(LinearRouteFamily::Fp8, width, fp8_wide_route_table(policy));
        },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) -> std::size_t {
            const auto& shape = resolve_shape(n, k, policy, table);
            if (!allows_a8(policy)) return 0;
            if (ada_crossover_shape(shape)) {
                std::size_t capacity = 0;
                for (std::int32_t width = first; width <= last; ++width) {
                    if (ada_uses_a8(shape, width))
                        capacity = std::max(capacity, fp8_a8_workspace_capacity_bytes(
                            width, k, shape.partial_capacity_bytes
                                          ? shape.partial_capacity_bytes(width) : 0));
                }
                return capacity;
            }
            if (!shape.uses_a8(first, last)) return 0;
            return fp8_a8_workspace_capacity_bytes(
                last, k, shape.partial_capacity_bytes ? shape.partial_capacity_bytes(last) : 0);
        });
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream) {
    validate_fp8_weight(weight, "fp8 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("fp8 linear: T must be positive");
    if (out.dtype == DType::FP32 || !fp8_registered(weight.n, weight.k)) {
        if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
        return fp8_general_a16(x, weight, out, stream);
    }
    const auto& shape =
        resolve_shape(weight.n, weight.k, policy,
                      linear_route_table(LinearRouteFamily::Fp8, x.ne[1],
                                         fp8_wide_route_table(policy)));
    if (!allows_a8(policy)) return shape.a16(x, weight, out, stream);
    const bool selected_a8 = ada_crossover_shape(shape)
                                 ? ada_uses_a8(shape, x.ne[1])
                                 : shape.uses_a8(x.ne[1], x.ne[1]);
    if (!selected_a8) return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("fp8 A8 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_fp8_a8_workspace(
        *workspace, x.ne[1], weight.k,
        shape.partial_capacity_bytes ? shape.partial_capacity_bytes(x.ne[1]) : 0);
    shape.a8(x, weight, out, scratch, stream);
}
} // namespace ninfer::ops::detail
