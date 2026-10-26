#pragma once

// FP8 A8 runs on Ada's ordinary E4M3 MMA or Blackwell's block-scaled MMA.
// Native NVFP4 A4 remains Blackwell-only. Guard entry points before launching a kernel whose
// unsupported device image contains a trap.

#include "ops/common/device_tier.h"

#include <stdexcept>

namespace ninfer::ops::detail {

inline void require_fp8_a8_card() {
    if (!current_device_fp8()) {
        throw std::runtime_error("FP8 A8 execution requires an sm_89 or sm_120a image");
    }
}

inline void require_nvfp4_a4_card() {
    if constexpr (kBuildTiers > 1) {
        if (!current_device_blackwell()) {
            throw std::runtime_error("NVFP4 A4 execution requires an sm_120a GPU");
        }
    }
}

} // namespace ninfer::ops::detail
