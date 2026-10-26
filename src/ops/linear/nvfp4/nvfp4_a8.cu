#include "ops/linear/nvfp4/nvfp4_a8_plan.h"
#include "core/device.h"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/common/token_slices.h"
#include "ops/linear/nvfp4/nvfp4_codec_unified.cuh"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"
#include <type_traits>

namespace ninfer::ops::detail {
namespace {
enum class Fusion { Linear, SwiGlu, Add };

// Rowwise affine FP8 activation approximation. Min/max, the overflow-safe
// midpoint (only for strictly one-sided rows), residual subtraction and scale
// are FP32. Zero-crossing rows retain center zero. E4M3 conversion is RN
// with finite saturation. Constant nonzero rows retain their center with zero
// scale/codes; the GEMM must still apply the center's weighted code-sum term.
template<int K>
__global__ __launch_bounds__(256) void nvfp4_center_quantize_kernel(
    const __nv_bfloat16* input,std::uint8_t* codes,float* scales,float* centers) {
    constexpr int Threads=256, Warps=Threads/32, Pairs=K/(Threads*2);
    static_assert(K%(Threads*2)==0);
    __shared__ float minima[Warps],maxima[Warps],center,scale;
    const int token=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const auto* source=reinterpret_cast<const unsigned*>(input+static_cast<std::int64_t>(token)*K);
    float2 values[Pairs];
    float lo=INFINITY,hi=-INFINITY;
#pragma unroll
    for(int item=0;item<Pairs;++item) {
        values[item]=bf16x2_bits_to_float2(source[tid+item*Threads]);
        lo=fminf(lo,fminf(values[item].x,values[item].y));
        hi=fmaxf(hi,fmaxf(values[item].x,values[item].y));
    }
#pragma unroll
    for(int delta=16;delta;delta>>=1) {
        lo=fminf(lo,__shfl_xor_sync(0xffffffff,lo,delta));
        hi=fmaxf(hi,__shfl_xor_sync(0xffffffff,hi,delta));
    }
    if(lane==0){minima[warp]=lo;maxima[warp]=hi;}
    __syncthreads();
    if(warp==0) {
        lo=lane<Warps?minima[lane]:INFINITY;
        hi=lane<Warps?maxima[lane]:-INFINITY;
#pragma unroll
        for(int delta=16;delta;delta>>=1) {
            lo=fminf(lo,__shfl_xor_sync(0xffffffff,lo,delta));
            hi=fmaxf(hi,__shfl_xor_sync(0xffffffff,hi,delta));
        }
        if(lane==0) {
            center=(lo>0.0F || hi<0.0F)
                ? __fadd_rn(__fmul_rn(lo,0.5F),__fmul_rn(hi,0.5F)) : 0.0F;
            const float radius=fmaxf(fabsf(__fsub_rn(lo,center)),fabsf(__fsub_rn(hi,center)));
            scale=radius>0.0F?__fdiv_rn(radius,448.0F):0.0F;
        }
    }
    __syncthreads();
    const float inverse=scale>0.0F?__fdiv_rn(1.0F,scale):0.0F;
    auto* destination=reinterpret_cast<std::uint16_t*>(codes+static_cast<std::int64_t>(token)*K);
#pragma unroll
    for(int item=0;item<Pairs;++item) {
        float2 residual=make_float2(__fsub_rn(values[item].x,center),__fsub_rn(values[item].y,center));
        if(isfinite(inverse)) {
            residual.x=__fmul_rn(residual.x,inverse);
            residual.y=__fmul_rn(residual.y,inverse);
        } else {
            // A finite BF16 residual can need a scale with an overflowing reciprocal.
            residual.x=__fdiv_rn(residual.x,scale);
            residual.y=__fdiv_rn(residual.y,scale);
        }
        destination[tid+item*Threads]=__nv_cvt_float2_to_fp8x2(residual,__NV_SATFINITE,__NV_E4M3);
    }
    if(tid==0){scales[token]=scale;centers[token]=center;}
}
template<int K>
void quantize_centered(const Tensor& x,Nvfp4A8Workspace scratch,cudaStream_t stream) {
    nvfp4_center_quantize_kernel<K><<<x.ne[1],256,0,stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),scratch.codes,scratch.scales,scratch.centers);
    CUDA_CHECK(cudaGetLastError());
}
void quantize_centered(const Tensor& x,Nvfp4A8Workspace scratch,cudaStream_t stream) {
    switch(x.ne[0]) {
    case 5120: return quantize_centered<5120>(x,scratch,stream);
    case 6144: return quantize_centered<6144>(x,scratch,stream);
    case 17408: return quantize_centered<17408>(x,scratch,stream);
    default: throw std::invalid_argument("nvfp4 A8 quantize: unsupported K");
    }
}

// Exact E2M1 values measured in half units, for the affine center correction.
__device__ __forceinline__ int e2m1_pair_half_sum(std::uint8_t pair) {
    constexpr std::uint64_t magnitudes=0x0c08060403020100ULL;
    int lo=(magnitudes>>((pair&7)*8))&255;
    int hi=(magnitudes>>(((pair>>4)&7)*8))&255;
    if(pair&8)lo=-lo;
    if(pair&128)hi=-hi;
    return lo+hi;
}

__device__ __forceinline__ std::uint16_t e2m1_pair_as_e4m3(std::uint8_t pair) {
    // Exact finite encodings for {0,.5,1,1.5,2,3,4,6}; preserve both signs,
    // including negative zero. This is the exact result of RN E4M3 conversion.
    constexpr std::uint64_t magnitudes = 0x4c4844403c383000ULL;
    const unsigned lo = ((magnitudes >> ((pair & 7) * 8)) & 255) | ((pair & 8) << 4);
    const unsigned hi = ((magnitudes >> (((pair >> 4) & 7) * 8)) & 255) | ((pair & 128));
    return static_cast<std::uint16_t>(lo | (hi << 8));
}

// A native G16 scale is restored in FP32 after its own dot product. E2M1 values
// themselves are exact E4M3 operands: no small source scale can underflow in the
// bridge and no large one can clip. Ada native k16 MMA matches each G16
// reduction without padded arithmetic.
// Only activation conversion adds an FP8 rounding boundary.
template <bool A8, Fusion F>
__global__ __launch_bounds__(128) void nvfp4_group_mma_kernel(
    const void* __restrict__ x, const float* __restrict__ activation_scales,
    const float* __restrict__ activation_centers,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const float* __restrict__ divisors, int divisor_rows, __nv_bfloat16* out,
    int rows, int k, int token_offset, int count) {
    constexpr int BM = 16, BN = 32, BK = 128;
    constexpr int Groups = BK / 16;
    using Element = std::conditional_t<A8, std::uint8_t, __nv_bfloat16>;
    __shared__ __align__(16) Element activation[BM * BK];
    __shared__ __align__(16) Element weight[BN * BK];
    __shared__ float group_scales[BN * Groups];
    __shared__ float code_sums[A8 ? BN * Groups : 1];
    __shared__ float values[BM * BN];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int row_begin = blockIdx.x * (F == Fusion::SwiGlu ? BN / 2 : BN);
    const int token_begin = token_offset + blockIdx.y * BM;
    const int tokens = token_offset + count;
    const auto parent = [&](int local) {
        if constexpr (F == Fusion::SwiGlu)
            return row_begin + local % (BN / 2) + (local >= BN / 2 ? rows / 2 : 0);
        else return row_begin + local;
    };
    // This block-wide decision is uniform, including predicated token tails.
    // Normalized zero-crossing rows need no affine correction, so skip its
    // weight-unpack reductions and weighted row-sum accumulation entirely.
    bool needs_center=false;
    if constexpr(A8) {
        const int token=token_begin+tid;
        needs_center=__syncthreads_or(tid<BM && token<tokens &&
                                     activation_centers[token]!=0.0F)!=0;
    }
    float acc[4] = {};
    float code_acc[2] = {}; // weighted E2M1 row sums, independent of token
    for (int tile = 0; tile < k / BK; ++tile) {
        // Stage a complete K128 tile once; eight G16 partials then reuse it
        // without a barrier between groups.
        const int local_token = tid / Groups, group_in_tile = tid % Groups;
        const int token = token_begin + local_token;
        const int input_col = tile * BK + group_in_tile * 16;
        if constexpr (A8) {
            const auto* source = static_cast<const std::uint8_t*>(x);
            const uint4 v = token < tokens
                ? load_vec<uint4>(source + static_cast<std::int64_t>(token) * k + input_col)
                : uint4{};
            store_vec(activation + local_token * BK + group_in_tile * 16, v);
        } else {
            const auto* source = static_cast<const __nv_bfloat16*>(x);
            const auto* src = source + static_cast<std::int64_t>(token < tokens ? token : 0) * k + input_col;
            store_vec(activation + local_token * BK + group_in_tile * 16, token < tokens ? load_vec<uint4>(src) : uint4{});
            store_vec(activation + local_token * BK + group_in_tile * 16 + 8, token < tokens ? load_vec<uint4>(src + 8) : uint4{});
        }
        for (int pair = tid; pair < BN * (BK / 2); pair += 128) {
            const int local = pair / (BK / 2), col = 2 * (pair % (BK / 2)), row = parent(local);
            const auto packed = codes[static_cast<std::int64_t>(row) * (k / 2) + tile * (BK / 2) + col / 2];
            if constexpr (A8) {
                *reinterpret_cast<std::uint16_t*>(weight + local * BK + col) = e2m1_pair_as_e4m3(packed);
                // Eight adjacent lanes own all pairs of one G16. Integer half
                // units make the code sum exact before its stored scale.
                if(needs_center) {
                int half_sum=e2m1_pair_half_sum(packed);
#pragma unroll
                for(int delta=4;delta;delta>>=1)
                    half_sum+=__shfl_xor_sync(0xffffffff,half_sum,delta,8);
                if((pair&7)==0)
                    code_sums[local*Groups+col/16]=static_cast<float>(half_sum)*0.5F;
                }
            } else {
                const float2 unpacked = unified::decode_nvfp4_e2m1x2(packed);
                *reinterpret_cast<__nv_bfloat162*>(weight + local * BK + col) =
                    __floats2bfloat162_rn(unpacked.x, unpacked.y);
            }
        }
        for (int item = tid; item < BN * Groups; item += 128) {
            const int local = item / Groups, group = item % Groups;
            group_scales[item] = unified::decode_nvfp4_e4m3(
                scales[unified::nvfp4_scale_byte_offset(parent(local), tile * Groups + group, k)]);
        }
        __syncthreads();
#pragma unroll
        for (int group = 0; group < Groups; ++group) {
        float part[4] = {};
        if constexpr (A8) {
            const int gid = lane >> 2, col = group * 16 + 4 * (lane & 3);
            const unsigned a0 = load_vec<unsigned>(activation + gid * BK + col);
            const unsigned a1 = load_vec<unsigned>(activation + (gid + 8) * BK + col);
            const unsigned b0 = load_vec<unsigned>(weight + (warp * 8 + gid) * BK + col);
            mma_fp8_e4m3_k16(part[0],part[1],part[2],part[3],a0,a1,b0);
        } else {
            const int arow = (lane & 7) + (((lane >> 3) & 1) * 8);
            const int acol = group * 16 + (lane >> 4) * 8;
            const int brow = warp * 8 + (lane & 7);
            const int bcol = group * 16 + ((lane >> 3) & 1) * 8;
            unsigned a0, a1, a2, a3, b0, b1;
            ldmatrix_x4(a0,a1,a2,a3,smem_addr(activation + arow * BK + acol));
            ldmatrix_x2(b0,b1,smem_addr(weight + brow * BK + bcol));
            mma_bf16(part[0],part[1],part[2],part[3],a0,a1,a2,a3,b0,b1);
        }
        const int col = warp * 8 + 2 * (lane & 3);
        acc[0] = __fmaf_rn(part[0],group_scales[col * Groups + group],acc[0]);
        acc[1] = __fmaf_rn(part[1],group_scales[(col+1) * Groups + group],acc[1]);
        acc[2] = __fmaf_rn(part[2],group_scales[col * Groups + group],acc[2]);
        acc[3] = __fmaf_rn(part[3],group_scales[(col+1) * Groups + group],acc[3]);
        if constexpr(A8) {
            if(needs_center) {
            code_acc[0]=__fmaf_rn(code_sums[col*Groups+group],group_scales[col*Groups+group],code_acc[0]);
            code_acc[1]=__fmaf_rn(code_sums[(col+1)*Groups+group],group_scales[(col+1)*Groups+group],code_acc[1]);
            }
        }
        }
        __syncthreads();
    }
    const int local_token = lane >> 2, local_row = warp * 8 + 2 * (lane & 3);
    const int row0 = parent(local_row), row1 = parent(local_row+1);
    const float divisor0 = divisors[row0 / divisor_rows];
    const float divisor1 = divisors[row1 / divisor_rows];
    const auto finish = [&](int local_t, int local_r, float sum, float code_sum, float divisor) {
        const int token = token_begin + local_t;
        if constexpr(A8) {
            const float as=token<tokens?activation_scales[token]:0.0F;
            const float center=token<tokens?activation_centers[token]:0.0F;
            const float ratio=as==0.0F?0.0F:__fdiv_rn(as,divisor);
            const float center_ratio=center==0.0F?0.0F:__fdiv_rn(center,divisor);
            const bool normal=(as==0.0F||(isfinite(ratio)&&fabsf(ratio)>=0x1p-126F)) &&
                (center==0.0F||(isfinite(center_ratio)&&fabsf(center_ratio)>=0x1p-126F));
            // Q and the exact weighted code sum are both FP32 reductions.
            // Divide both affine coefficients by this row's global divisor,
            // then combine with one FMA. Zero scale does not erase a center.
            // Rare exponent-range failures use only this scalar FP64 rescale;
            // no GEMM accumulation or stored operand is widened.
            float value=normal
                ? __fmaf_rn(sum,ratio,__fmul_rn(code_sum,center_ratio))
                : static_cast<float>((static_cast<double>(sum)*as+
                                      static_cast<double>(code_sum)*center)/divisor);
            if constexpr(F==Fusion::SwiGlu) {
                values[local_t*BN+local_r]=value;
            } else if(token<tokens) {
                const int row=parent(local_r);
                if constexpr(F==Fusion::Add) {
                    const Nvfp4AddResidualEpilogue epilogue{out,rows};
                    value=normal
                        ? __fmaf_rn(sum,ratio,epilogue.apply_scaled(row,token,code_sum,center_ratio))
                        : epilogue.apply(row,token,value);
                }
                Nvfp4ContiguousOutput{out,rows}.store(row,token,value);
            }
            return;
        }
        // Preserve the existing A16 arithmetic and finalization below.
        const float as = A8 ? (token < tokens ? activation_scales[token] : 0.0F) : 1.0F;
        const float ratio = as == 0.0F ? 0.0F : __fdiv_rn(as,divisor);
        // The normal route stays FP32. A rare exponent-range guard keeps
        // finite global divisors from causing a spurious 0*Inf or premature
        // underflow before a finite result can be formed.
        const bool normal = as == 0.0F || (isfinite(ratio) && fabsf(ratio) >= 0x1p-126F);
        const float value = as == 0.0F ? 0.0F : normal
            ? __fmul_rn(sum,ratio)
            : static_cast<float>((static_cast<double>(sum)*static_cast<double>(as))/static_cast<double>(divisor));
        if constexpr (F == Fusion::SwiGlu)
            values[local_t * BN + local_r] = value;
        else if (token < tokens) {
            const int row = parent(local_r);
            const Nvfp4ContiguousOutput destination{out,rows};
            if constexpr (F == Fusion::Add) {
                const Nvfp4AddResidualEpilogue epilogue{out,rows};
                destination.store(row,token,normal
                    ? epilogue.apply_scaled(row,token,sum,ratio)
                    : epilogue.apply(row,token,value));
            } else destination.store(row,token,value);
        }
    };
    finish(local_token,local_row,acc[0],code_acc[0],divisor0);
    finish(local_token,local_row+1,acc[1],code_acc[1],divisor1);
    finish(local_token+8,local_row,acc[2],code_acc[0],divisor0);
    finish(local_token+8,local_row+1,acc[3],code_acc[1],divisor1);
    if constexpr (F == Fusion::SwiGlu) {
        __syncthreads();
        for (int item = tid; item < BM * (BN / 2); item += 128) {
            const int lt = item / (BN / 2), lr = item % (BN / 2), token = token_begin + lt;
            if (token < tokens)
                unified::Nvfp4SwiGluEpilogue{}.apply_pair(
                    Nvfp4ContiguousOutput{out,rows/2},row_begin+lr,token,
                    values[lt*BN+lr],values[lt*BN+lr+BN/2]);
        }
    }
}

template <bool A8, Fusion F>
void launch_group(const Tensor& x, const Weight& w, Tensor& out,
                  Nvfp4A8Workspace scratch, cudaStream_t stream) {
    for_each_token_slice(x.ne[1],16,[&](int offset,int count) {
        const dim3 grid(w.n / 32,(count+15)/16);
        nvfp4_group_mma_kernel<A8,F><<<grid,128,0,stream>>>(
            A8 ? static_cast<const void*>(scratch.codes) : x.data,scratch.scales,scratch.centers,
            static_cast<const std::uint8_t*>(w.qdata),static_cast<const std::uint8_t*>(w.scales),
            static_cast<const float*>(w.weight_divisors),w.weight_divisor_rows,static_cast<__nv_bfloat16*>(out.data),
            w.n,w.k,offset,count);
        CUDA_CHECK(cudaGetLastError());
    });
}

template <Fusion F>
void launch_a8(const Tensor& x,const Weight& w,Tensor& out,WorkspaceArena& workspace,cudaStream_t stream) {
    if (!current_device_ada_fp8()) throw std::runtime_error("NVFP4 A8 requires an sm_89 image");
    auto scope = workspace.scope();
    const auto scratch = allocate_nvfp4_a8_workspace(workspace,x.ne[1],w.k);
    quantize_centered(x,scratch,stream);
    launch_group<true,F>(x,w,out,scratch,stream);
}
} // namespace
void nvfp4_a8_linear_launch(const Tensor& x,const Weight& w,Tensor& y,WorkspaceArena& a,cudaStream_t s) { launch_a8<Fusion::Linear>(x,w,y,a,s); }
void nvfp4_a8_swiglu_launch(const Tensor& x,const Weight& w,Tensor& y,WorkspaceArena& a,cudaStream_t s) { launch_a8<Fusion::SwiGlu>(x,w,y,a,s); }
void nvfp4_a8_add_launch(const Tensor& x,const Weight& w,Tensor& y,WorkspaceArena& a,cudaStream_t s) { launch_a8<Fusion::Add>(x,w,y,a,s); }
} // namespace ninfer::ops::detail
