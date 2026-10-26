#pragma once
#include "ops/common/math.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail::unified {
template <int RowsPerBranch>
struct Nvfp4SwiGluRows {
    static constexpr bool kPaired = true;

    __device__ __forceinline__ int weight_row(int begin, int row, int rows) const {
        return begin + row % RowsPerBranch + (row >= RowsPerBranch ? rows / 2 : 0);
    }
};

struct Nvfp4SwiGluEpilogue {
    template <class Output>
    __device__ __forceinline__ void apply_pair(Output output, int row, int token, float gate,
                                               float up) const {
        output.store(row, token, silu(gate) * up);
    }
};

// Row-major A16 MMA leaves gate/up in FP32 until this complete fused
// operation. Reuse its mainloop shared memory after the final MMA; no global
// projection tensor, intermediate BF16 rounding, or expanded weights exist.
struct Nvfp4SwiGluMmaEpilogue {
    const float* divisors;
    int divisor_rows;

    template <class S>
    static constexpr int kSharedBytes = S::kBlockRows * S::kBlockTokens * sizeof(float);

    template <class S, bool Full, class Output>
    __device__ __forceinline__ void finish_tile(
        Output output, unsigned char* scratch, float (&acc)[S::kMmaRows][S::kMmaTokens][4],
        int row_begin, int token_begin, int rows, int token_end) const {
        constexpr int BM=S::kBlockRows, BN=S::kBlockTokens, half=BM/2;
        const int tid=static_cast<int>(threadIdx.x), warp=tid>>5, lane=tid&31;
        const int wm=warp/S::kWarpsTokens, wn=warp%S::kWarpsTokens;
        auto* tile=reinterpret_cast<float*>(scratch);
        __syncthreads();
#pragma unroll
        for (int mi=0; mi<S::kMmaRows; ++mi) {
            const int row=wm*S::kWarpRows+mi*16+(lane>>2);
#pragma unroll
            for (int ni=0; ni<S::kMmaTokens; ++ni) {
                const int token=wn*S::kWarpTokens+ni*8+2*(lane&3);
                tile[token*BM+row]=acc[mi][ni][0];
                tile[(token+1)*BM+row]=acc[mi][ni][1];
                tile[token*BM+row+8]=acc[mi][ni][2];
                tile[(token+1)*BM+row+8]=acc[mi][ni][3];
            }
        }
        __syncthreads();
        for (int item=tid; item<BN*half; item+=S::kThreads) {
            const int token=item/half, local=item%half, row=row_begin+local;
            if (Full || token_begin+token<token_end) {
                const float gate=__fdiv_rn(tile[token*BM+local],divisors[row/divisor_rows]);
                const float up=__fdiv_rn(tile[token*BM+local+half],divisors[(row+rows/2)/divisor_rows]);
                output.store(row,token_begin+token,silu(gate)*up);
            }
        }
    }
};
} // namespace ninfer::ops::detail::unified
