#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_shapes_unified.h"
#include "ops/linear/nvfp4/nvfp4_launch_unified.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
template<int Rows>
void head_a16(const Tensor& x,const Weight& w,Tensor& y,cudaStream_t stream) {
    using G = Nvfp4Geometry<Rows,5120>;
    using Gemv = unified::Nvfp4A16GemvSchedule<8,2,16,4,unified::Nvfp4ScaleAccess::StagedRaw,unified::Nvfp4CodeCache::Default,2>;
    if (x.ne[1] == 1) return unified::nvfp4_linear_a16_gemv<G,Gemv>(x,w,y,stream);
    if (x.ne[1] <= 8) return unified::nvfp4_linear_a16_sliced_k<G,unified::Nvfp4SlicedInstance<8,8,2>>(x,w,y,stream);
    if (x.ne[1] <= 16) return unified::nvfp4_linear_a16_sliced_k<G,unified::Nvfp4SlicedInstance<16,8,2>>(x,w,y,stream);
    if (x.ne[1] <= 64) return unified::nvfp4_linear_a16_mma<G,unified::Nvfp4A16MmaSchedule<32,64,64,16,16,1,2>>(x,w,y,stream);
    unified::nvfp4_linear_a16_mma<G,unified::Nvfp4A16MmaSchedule<64,128,64,64,16,2,2>>(x,w,y,stream);
}
bool head_uses_a4(std::int32_t,std::int32_t) { return false; }
}
const Nvfp4LinearShape kNvfp4N248320K5120{248320,5120,head_a16<248320>,nullptr,head_uses_a4};
const Nvfp4LinearShape kNvfp4N131072K5120{131072,5120,head_a16<131072>,nullptr,head_uses_a4};
namespace unified {
const Nvfp4LinearShape kNvfp4N248320K5120{248320,5120,head_a16<248320>,nullptr,head_uses_a4};
const Nvfp4LinearShape kNvfp4N131072K5120{131072,5120,head_a16<131072>,nullptr,head_uses_a4};
}
}
