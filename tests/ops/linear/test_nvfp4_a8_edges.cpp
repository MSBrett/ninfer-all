#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/common/device_route.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {
using namespace ninfer;
namespace qw = test::quantized_weight;
enum class Op { Linear, SwiGlu, Add };
constexpr int kTokens = 19;
constexpr auto kPolicy = ops::LinearPolicy::AllowA8;

std::size_t scale_byte(const qw::PackedWeight& p,int row,int group) {
    // Test-owned transcription of the stored format, independent of CUDA codecs.
    return p.scale_plane_offset + static_cast<std::size_t>((row/128)*(p.weight.k/64)+group/4)*512
        + (row%32)*16 + ((row%128)/32)*4 + group%4;
}
qw::PackedWeight edge_weight(int rows,int k,bool stacked) {
    qw::PatternedWeightOptions options;
    options.weight_scale_divisor = 3.0F;
    options.input_scale_divisor = 7.0F;
    auto p = qw::make_patterned_weight(QType::NVFP4,rows,k,29,options);
    std::fill(p.payload.begin(),p.payload.begin()+p.weight_divisor_offset,0);
    // First 127 rows cover every nonnegative finite E4M3 scale encoding,
    // including zero, 2^-9, the normal/subnormal seam, and 448.
    // Every E2M1 code (both zero signs included) appears in each scale sweep.
    for (int row=0;row<rows;++row) {
        for (int branch=0;branch<2;++branch) {
            const int group = branch == 0 ? 0 : k/16-1;
            p.payload[scale_byte(p,row,group)] = static_cast<std::uint8_t>(
                branch == 0 ? row%127 : (11*row+7)%127);
            for (int col=0;col<16;col+=2) {
                auto lo=static_cast<std::uint8_t>((row+col+branch*3)&15);
                auto hi=static_cast<std::uint8_t>((row+col+1+branch*3)&15);
                // Break each balanced code cycle in different positions. The
                // weighted center correction is deliberately nonzero, and
                // differing G16 scales prohibit an unweighted sum shortcut.
                if(branch==0 && col==14) hi=7;
                if(branch==1 && col==14) lo=15;
                p.payload[static_cast<std::size_t>(row)*k/2+group*8+col/2] = lo|(hi<<4);
            }
        }
    }
    const float first = 3.0F, second = 5.0F;
    p.payload.resize(p.weight_divisor_offset+(stacked?2:1)*sizeof(float));
    std::memcpy(p.payload.data()+p.weight_divisor_offset,&first,sizeof(first));
    if (stacked) std::memcpy(p.payload.data()+p.weight_divisor_offset+sizeof(float),&second,sizeof(second));
    p.weight.payload_bytes=p.payload.size();
    p.weight.weight_divisor_rows=stacked?rows/2:rows;
    return p;
}
std::vector<std::uint16_t> input(int k,int revision) {
    constexpr std::array<int,7> exponents{-100,-40,-4,0,4,16,30};
    constexpr std::array<float,16> mantissas{448,-448,224,-224,112,-112,56,-56,28,-28,14,-14,7,-7,0,0};
    std::vector<std::uint16_t> result(static_cast<std::size_t>(k)*kTokens);
    for(int token=0;token<kTokens;++token) {
        const int exponent=exponents[(token+revision)%exponents.size()];
        for(int col=0;col<k;++col) {
            // This directed fixture keeps residuals exactly E4M3 after the
            // row midpoint. Separate randomized suites retain the public A8
            // budget against the original, unquantized BF16-input oracle.
            const float residual=mantissas[(col+revision*3)%16];
            const int variant=(token+revision)%4;
            const float represented=variant==1?4096.0F:
                variant==2?4096.0F+residual:residual;
            float value=std::ldexp(represented,exponent);
            if ((token+revision)%3==1) value=-value;
            if (token==0) value=0; // zero, constant nonzero and large-bias rows all coexist
            result[static_cast<std::size_t>(token)*k+col]=test::f32_to_bf16(value);
        }
    }
    return result;
}
std::vector<std::uint16_t> residual(int rows,int revision) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows)*kTokens);
    for(int t=0;t<kTokens;++t)
        for(int r=0;r<rows;++r)
            result[static_cast<std::size_t>(t)*rows+r]=test::f32_to_bf16(
                std::ldexp(static_cast<float>((r+revision)%7-3), t==0?0:(t+revision)%17-8));
    return result;
}
struct ProjectionOracle {
    double value=0,abs_products=0,abs_weights=0;
    int nonzero_weights=0;
};
ProjectionOracle dot(const qw::PackedWeight& weight,const std::vector<std::uint16_t>& x,int row,int token) {
    ProjectionOracle result;
    for(int k=0;k<weight.weight.k;++k) {
        const double w=qw::logical_weight_fp64(weight,row,k);
        const double product=w*static_cast<double>(
            test::bf16_to_f32(x[static_cast<std::size_t>(token)*weight.weight.k+k]));
        result.value+=product;
        result.abs_products+=std::abs(product);
        result.abs_weights+=std::abs(w);
        result.nonzero_weights+=w!=0;
    }
    return result;
}
double projection_error(const ProjectionOracle& p,double max_input) {
    // This fixture's residuals are exactly FP8, but private FP32 evaluation
    // need not cancel exactly. A midpoint obeys |center| <= max|x|, so the
    // absolute sum of both affine terms is bounded independently by
    // sum|w*x| + 2*max|x|*sum|w|. Four rounded operations per nonzero product
    // plus 16 for coefficient scaling/finalization conservatively cover the
    // two reductions and affine rescale: gamma_n = n*u/(1-n*u), u=2^-24.
    // Zero products add exactly, so sparse zero padding adds no error budget.
    constexpr double u=0x1p-24;
    const double nu=(4.0*p.nonzero_weights+16.0)*u;
    return nu/(1.0-nu)*(p.abs_products+2.0*max_input*p.abs_weights);
}
int check(Op op,const qw::PackedWeight& weight,const std::vector<std::uint16_t>& x,
          const std::vector<std::uint16_t>& initial,const std::vector<std::uint16_t>& actual,
          int output_rows,const char* mode,bool zero_weight=false) {
    int failures=0;
    std::vector<int> rows;
    for(int row=0;row<256;++row) rows.push_back(row);
    for(int row : {output_rows/2-1,output_rows/2,output_rows-129,output_rows-128,output_rows-1})
        rows.push_back(row);
    std::array<double,kTokens> max_input{};
    for(int t=0;t<kTokens;++t) for(int k=0;k<weight.weight.k;++k)
        max_input[t]=std::max(max_input[t],std::abs(static_cast<double>(
            test::bf16_to_f32(x[static_cast<std::size_t>(t)*weight.weight.k+k]))));
    for(int t=0;t<kTokens;++t) for(int row:rows) {
        const ProjectionOracle projection=zero_weight?ProjectionOracle{}:dot(weight,x,row,t);
        double expected=projection.value;
        double arithmetic_error=projection_error(projection,max_input[t]);
        double tail_underflow=0;
        if(op==Op::SwiGlu) {
            const ProjectionOracle other=zero_weight?ProjectionOracle{}:dot(weight,x,row+output_rows,t);
            const double gate=expected,up=other.value;
            const double gate_error=arithmetic_error,up_error=projection_error(other,max_input[t]);
            const double activated=gate/(1.0+std::exp(-gate));
            expected=activated*up;
            // |SiLU'| <= 1+1/e < 1.5 globally. For a negative interval use
            // the tighter e^(g+dg)*(1+|g|+dg), retaining tiny-tail sensitivity.
            const double upper_gate=gate+gate_error;
            const double derivative=upper_gate<0
                ? std::min(1.5,std::exp(upper_gate)*(1.0+std::abs(gate)+gate_error)) : 1.5;
            arithmetic_error=derivative*gate_error*(std::abs(up)+up_error)+
                std::abs(activated)*up_error;
            // Existing FP32 exp/reciprocal SiLU may overflow exp(-g) or
            // flush a subnormal sigmoid. Only below the FP32 normal range,
            // |SiLU(g)*up| <= (|g|+dg)*(|up|+du)*FLT_MIN covers that loss.
            const double min_normal=std::numeric_limits<float>::min();
            if(upper_gate<std::log(min_normal))
                tail_underflow=(std::abs(gate)+gate_error)*(std::abs(up)+up_error)*min_normal;
        } else if(op==Op::Add) {
            const double initial_value=test::bf16_to_f32(initial[static_cast<std::size_t>(t)*output_rows+row]);
            arithmetic_error+=0x1p-24*(std::abs(expected)+std::abs(initial_value));
            expected+=initial_value;
        }
        const double observed=test::bf16_to_f32(actual[static_cast<std::size_t>(t)*output_rows+row]);
        // Keep the existing 0.006 relative allowance. Replace the invalid
        // exact-cancellation/blanket 2e-38 assumptions with scale-sensitive
        // FP32 forward error, the specific SiLU tail loss, and half a BF16
        // subnormal ULP. The public A8 aggregate/gross suites are unchanged.
        const double tolerance=0.006*std::abs(expected)+arithmetic_error+
            tail_underflow+std::ldexp(1.0,-134);
        if(!std::isfinite(observed) || std::abs(observed-expected)>tolerance) {
            if(failures<8) std::cerr<<"NVFP4 A8 edge "<<mode<<" op="<<static_cast<int>(op)
                <<" row="<<row<<" token="<<t<<" got="<<observed<<" expected="<<expected
                <<" fp32_bound="<<arithmetic_error<<" tail_bound="<<tail_underflow<<'\n';
            ++failures;
        }
    }
    return failures;
}
int run(Op op) {
    // DeviceRouteForce owns one active key. Scope it per Op so each directed
    // invocation measures the candidate instead of the production A16 choice.
    const ops::DeviceRouteForce candidate(
        op==Op::Linear ? "nvfp4_linear/14336x5120" :
        op==Op::SwiGlu ? "nvfp4_linear_swiglu/34816x5120" : "nvfp4_linear_add/5120x6144","a8");
    const int n=op==Op::SwiGlu?34816:(op==Op::Add?5120:14336);
    const int k=op==Op::Add?6144:5120;
    const int output_rows=op==Op::SwiGlu?n/2:n;
    auto packed=edge_weight(n,k,op==Op::SwiGlu);
    DeviceBuffer dw(packed.payload.size());
    dw.copy_from_host(packed.payload.data(),packed.payload.size());
    const Weight weight=packed.device_weight(dw.p);
    DeviceBuffer dx(static_cast<std::size_t>(k)*kTokens*2);
    DeviceBuffer dy(static_cast<std::size_t>(output_rows)*kTokens*2);
    Tensor tx(dx.p,DType::BF16,{k,kTokens}),ty(dy.p,DType::BF16,{output_rows,kTokens});
    const std::size_t bytes=op==Op::Linear
        ? ops::linear_workspace_capacity_bytes(QType::NVFP4,n,k,kPolicy,kTokens,kTokens)
        : op==Op::SwiGlu
          ? ops::linear_swiglu_workspace_capacity_bytes(QType::NVFP4,n,k,kPolicy,kTokens,kTokens)
          : ops::linear_add_workspace_capacity_bytes(QType::NVFP4,n,k,kPolicy,kTokens,kTokens);
    WorkspaceArena workspace(std::max<std::size_t>(bytes,256));
    cudaStream_t stream=nullptr;
    test::cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream");
    const auto invoke=[&] {
        if(op==Op::Linear) ops::linear(tx,weight,ty,kPolicy,workspace,stream);
        else if(op==Op::SwiGlu) ops::linear_swiglu(tx,weight,ty,kPolicy,workspace,stream);
        else ops::linear_add(tx,weight,ty,kPolicy,workspace,stream);
    };
    auto x=input(k,0),initial=residual(output_rows,0);
    dx.copy_from_host(x.data(),dx.bytes);dy.copy_from_host(initial.data(),dy.bytes);
    // First CUDA invocation is captured; replay must consume changed live inputs.
    test::cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal),"begin capture");
    invoke();
    cudaGraph_t graph=nullptr;
    test::cuda_check(cudaStreamEndCapture(stream,&graph),"end capture");
    cudaGraphExec_t exec=nullptr;
    test::cuda_check(cudaGraphInstantiate(&exec,graph,0),"instantiate graph");
    int failures=0;
    for(int revision=0;revision<3;++revision) {
        if (revision==2) {
            // Zero stored codes with extreme, finite global divisors must
            // remain zero even when as/divisor overflows or is subnormal.
            std::fill(packed.payload.begin(),packed.payload.begin()+packed.code_plane_bytes,0);
            const float tiny=std::numeric_limits<float>::denorm_min();
            const float huge=std::numeric_limits<float>::max();
            std::memcpy(packed.payload.data()+packed.weight_divisor_offset,&tiny,sizeof(tiny));
            if(op==Op::SwiGlu)
                std::memcpy(packed.payload.data()+packed.weight_divisor_offset+sizeof(float),&huge,sizeof(huge));
            dw.copy_from_host(packed.payload.data(),packed.payload.size());
        }
        x=input(k,revision);initial=residual(output_rows,revision);
        dx.copy_from_host(x.data(),dx.bytes);dy.copy_from_host(initial.data(),dy.bytes);
        test::cuda_check(cudaGraphLaunch(exec,stream),"launch graph");
        test::cuda_synchronize(stream);
        std::vector<std::uint16_t> captured(static_cast<std::size_t>(output_rows)*kTokens);
        dy.copy_to_host(captured.data(),dy.bytes);
        failures+=check(op,packed,x,initial,captured,output_rows,"graph",revision==2);
        dy.copy_from_host(initial.data(),dy.bytes);
        invoke();test::cuda_synchronize(stream);
        std::vector<std::uint16_t> eager(captured.size());
        dy.copy_to_host(eager.data(),dy.bytes);
        failures+=check(op,packed,x,initial,eager,output_rows,"eager",revision==2);
        if(captured!=eager) { std::cerr<<"NVFP4 A8 eager/graph mismatch\n";++failures; }
    }
    test::cuda_check(cudaGraphExecDestroy(exec),"destroy graph exec");
    test::cuda_check(cudaGraphDestroy(graph),"destroy graph");
    test::cuda_check(cudaStreamDestroy(stream),"destroy stream");
    return failures;
}
}
int main() {
    if(ninfer::test::cuda_unavailable() || !ninfer::ops::current_device_ada_fp8()) {
        std::cout<<"SKIP: NVFP4 A8 edge qualification requires loaded Ada FP8 code\n";
        return 77;
    }
    try {
        int failures=run(Op::Linear)+run(Op::SwiGlu)+run(Op::Add);
        std::cout<<(failures?"FAIL":"OK")<<" NVFP4 A8 scale edges, nonzero code sums, affine constant/bias rows, changed-input graph replay\n";
        return failures?1:0;
    } catch(const std::exception& e) {std::cerr<<"NVFP4 A8 edges: "<<e.what()<<'\n';return 1;}
}
