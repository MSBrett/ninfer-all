#include "core/device.h"
#include "core/decode_graph.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/common/device_tier.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <string>

namespace {
using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;
namespace qw=quantized_weight;
constexpr int D=5120,N=14336;
constexpr float eps=1.0e-6F;
constexpr auto policy=ops::LinearPolicy::AllowA8;
constexpr std::array<int,8> columns{0,1,15,16,257,1023,4096,5119};
constexpr ReductionCriterion criterion{0.04,1.0/256.0,0.06};

struct TableScope {
    explicit TableScope(ops::detail::LinearRouteTable table) { ops::detail::force_linear_route_table(table); }
    ~TableScope() { ops::detail::force_linear_route_table(std::nullopt); }
};

qw::PackedWeight make_weight() {
    auto p=qw::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16,N,D,991U);
    std::fill(p.payload.begin(),p.payload.end(),0);
    constexpr std::array<std::uint8_t,8> codes{0x38,0xb0,0x40,0x28,0x34,0x44,0x30,0xb8};
    for (int row=0;row<N;++row) {
        for (int i=0;i<8;++i)
            p.payload[static_cast<std::size_t>(row)*D+columns[i]]=codes[(i+row)%8]^(row%3==0 ? 0x80U : 0U);
        qw::detail::store_u16_le(p.payload,p.scale_plane_offset+static_cast<std::size_t>(row)*2,
            f32_to_bf16((1.0F+static_cast<float>(row%7)/8.0F)/32.0F));
    }
    return p;
}

struct Inputs {
    std::vector<std::uint16_t> x,residual,gain,sum,norm;
};

Inputs inputs(int tokens,int revision,bool residual,bool read_alias) {
    Inputs h;
    h.x.resize(static_cast<std::size_t>(D)*tokens);
    h.residual.resize(h.x.size());h.sum.resize(h.x.size());h.norm.resize(h.x.size());h.gain.resize(D);
    for (int col=0;col<D;++col)
        h.gain[col]=f32_to_bf16(static_cast<float>((col*13+revision*3)%7-3)/8.0F);
    for (int t=0;t<tokens;++t) {
        double squares=0;
        for (int col=0;col<D;++col) {
            const int kind=(t+revision)%6;
            const float ramp=static_cast<float>((col+7*t+revision)%9-4)/8.0F;
            const float x=kind==0 ? 0.0F : kind==1 ? 0.5F*(revision+1) : kind==2 ? -0.5F :
                          kind==3 ? 2.0F+ramp : kind==4 ? ramp : ramp/4096.0F;
            const std::size_t i=static_cast<std::size_t>(t)*D+col;
            h.x[i]=f32_to_bf16(x);
            h.residual[i]=read_alias ? h.x[i] : f32_to_bf16(kind==0 ? 0.0F :
                static_cast<float>((col+t+revision)%5-2)/32.0F);
            // Optional residual addition is a represented BF16 boundary.
            h.sum[i]=residual ? f32_to_bf16(bf16_to_f32(h.x[i])+bf16_to_f32(h.residual[i])) : h.x[i];
            const double value=bf16_to_f32(h.sum[i]);
            squares+=value*value;
        }
        const double inv=1.0/std::sqrt(squares/D+static_cast<double>(eps));
        for (int col=0;col<D;++col) {
            const std::size_t i=static_cast<std::size_t>(t)*D+col;
            // Independent FP64 RMSNorm, then its declared BF16 boundary.
            h.norm[i]=f32_to_bf16(static_cast<float>(bf16_to_f32(h.sum[i])*inv*(1.0+bf16_to_f32(h.gain[col]))));
        }
    }
    return h;
}

int compare(const std::string& label,const GuardedBf16Tensor& actual,const GuardedBf16Tensor& reference) {
    if (actual.bits()==reference.bits()) return 0;
    std::cerr<<label<<": fused/unfused bits differ\n";return 1;
}

int compare_bytes(const std::string& label,const void* actual,const void* reference,std::size_t bytes) {
    const auto got=from_device<std::uint8_t>(actual,bytes);
    const auto expected=from_device<std::uint8_t>(reference,bytes);
    if (got==expected) return 0;
    const auto mismatch=std::mismatch(got.begin(),got.end(),expected.begin());
    std::cerr<<label<<": quantized byte mismatch at "<<(mismatch.first-got.begin())<<'\n';
    return 1;
}

int oracle(const std::string& label,const GuardedBf16Tensor& output,const qw::PackedWeight& w,
           const Inputs& h,int parent_begin,int rows,int tokens) {
    const auto bits=output.bits();
    std::vector<double> got,expected;
    for (int row : sampled_rows(rows,31)) for (int t=0;t<tokens;++t) {
        double sum=0;
        for (int col : columns)
            sum+=qw::logical_weight_fp64(w,parent_begin+row,col)*
                 bf16_to_f32(h.norm[static_cast<std::size_t>(t)*D+col]);
        got.push_back(bf16_to_f32(bits[static_cast<std::size_t>(t)*rows+row]));
        expected.push_back(sum);
    }
    return verify_reduction(label,got,expected,criterion);
}

int run_case(const DevicePackedWeight& parent,int tokens,bool residual,bool read_alias=false,bool store_sum=true) {
    const TableScope table(tokens<=17 ? ops::detail::LinearRouteTable::Legacy : ops::detail::LinearRouteTable::Unified);
    const Weight w=parent.view();
    if (!ops::attn_input_proj_fused_rmsnorm_fp8_eligible(w,policy,tokens)) {
        std::cerr<<"Expected eligible fused RMSNorm FP8 route T="<<tokens<<'\n';return 1;
    }
    GuardedBf16Tensor input(D,tokens),res(D,tokens),gain(D,1),sum(D,tokens),normalized(D,tokens),fused_sum(D,tokens);
    GuardedBf16Tensor fq(6144,tokens),fg(6144,tokens),fk(1024,tokens),fv(1024,tokens);
    GuardedBf16Tensor uq(6144,tokens),ug(6144,tokens),uk(1024,tokens),uv(1024,tokens);
    Tensor x=input.tensor(),r=read_alias ? x : res.tensor(),g=gain.tensor(),s=sum.tensor(),norm=normalized.tensor(),fs=fused_sum.tensor();
    Tensor q=fq.tensor(),gate=fg.tensor(),key=fk.tensor(),value=fv.tensor();
    Tensor rq=uq.tensor(),rg=ug.tensor(),rk=uk.tensor(),rv=uv.tensor();
    const auto capacity=ops::attn_input_proj_workspace_capacity_bytes(w.qtype,N,D,policy,tokens,tokens);
    GuardedDeviceBuffer a(capacity),b(capacity);
    WorkspaceArena fused_workspace(DeviceSpan{a.data(),capacity}),unfused_workspace(DeviceSpan{b.data(),capacity});
    const auto quant_capacity=ops::detail::fp8_a8_workspace_capacity_bytes(tokens,D);
    GuardedDeviceBuffer qa(quant_capacity),qb(quant_capacity);
    WorkspaceArena producer_workspace(DeviceSpan{qa.data(),quant_capacity}),
                   quantizer_workspace(DeviceSpan{qb.data(),quant_capacity});
    DeviceContext device;
    const auto fused=[&] {
        ops::attn_input_proj_fused_rmsnorm_fp8(x,g,eps,w,q,gate,key,value,policy,fused_workspace,
                                              device.stream,residual ? &r : nullptr,residual && store_sum ? &fs : nullptr);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    definition.capture(device.stream,fused);graph.instantiate(definition);
    int failures=0;
    for (int revision=0;revision<3;++revision) {
        const auto h=inputs(tokens,revision,residual,read_alias);
        input.copy_from_bits(h.x);res.copy_from_bits(h.residual);gain.copy_from_bits(h.gain);
        for (auto* output : {&fq,&fg,&fk,&fv,&fused_sum}) output->repaint(device.stream);
        graph.launch(device.stream);cuda_synchronize(device.stream);
        const std::string label="fused RMSNorm FP8 T="+std::to_string(tokens)+
            " residual="+std::to_string(residual)+" replay="+std::to_string(revision);
        // The unfused GPU pipeline is only the preservation oracle. Its output
        // never substitutes for the independent mathematical checks below.
        Tensor* norm_input=&x;
        if (residual) {
            sum.copy_from_bits(read_alias ? h.x : h.residual);
            ops::residual_add(x,s,device.stream);norm_input=&s;
        }
        ops::rmsnorm(*norm_input,g,eps,true,norm,device.stream);
        ops::attn_input_proj(norm,w,rq,rg,rk,rv,policy,unfused_workspace,device.stream);
        cuda_synchronize(device.stream);
        // Compare every input coordinate, not just those reached by this
        // sparse projection fixture. Only initialized code/scale/center
        // prefixes are contractual; aligned padding bytes are excluded.
        qa.fill(0xa5);qb.fill(0x5a);
        {
            auto producer_scope=producer_workspace.scope();
            auto quantizer_scope=quantizer_workspace.scope();
            const auto produced=ops::detail::allocate_fp8_a8_workspace(producer_workspace,tokens,D);
            const auto quantized=ops::detail::allocate_fp8_a8_workspace(quantizer_workspace,tokens,D);
            ops::detail::launch_fp8_a8_fused_rmsnorm(x,g,eps,produced,device.stream,
                residual ? &r : nullptr,residual && store_sum ? &fs : nullptr);
            ops::detail::launch_fp8_a8_quantize(norm,w,quantized,device.stream);
            cuda_synchronize(device.stream);
            failures+=compare_bytes(label+" all codes",produced.codes,quantized.codes,std::size_t(tokens)*D);
            failures+=compare_bytes(label+" all scales",produced.scales,quantized.scales,std::size_t(tokens)*sizeof(float));
            failures+=compare_bytes(label+" all centers",produced.centers,quantized.centers,std::size_t(tokens)*sizeof(float));
        }
        failures+=compare(label+" q",fq,uq)+compare(label+" gate",fg,ug)+
                  compare(label+" k",fk,uk)+compare(label+" v",fv,uv);
        if (residual && store_sum) {
            failures+=compare(label+" residual sum",fused_sum,sum);
            if (fused_sum.bits()!=h.sum) { std::cerr<<label<<": BF16 residual oracle mismatch\n";++failures; }
        }
        failures+=oracle(label+" q",fq,parent.host,h,0,6144,tokens);
        failures+=oracle(label+" k",fk,parent.host,h,6144,1024,tokens);
        failures+=oracle(label+" gate",fg,parent.host,h,7168,6144,tokens);
        failures+=oracle(label+" v",fv,parent.host,h,13312,1024,tokens);
        // Eager consumes the same changed live operands as the captured graph.
        const std::array saved{fq.bits(),fg.bits(),fk.bits(),fv.bits()};
        fused();cuda_synchronize(device.stream);
        if (saved!=std::array{fq.bits(),fg.bits(),fk.bits(),fv.bits()}) {
            std::cerr<<label<<": eager/graph mismatch\n";++failures;
        }
        if (input.bits()!=h.x || gain.bits()!=h.gain || (!read_alias && res.bits()!=h.residual)) {
            std::cerr<<label<<": input modified\n";++failures;
        }
        for (auto* output : {&input,&res,&gain,&sum,&normalized,&fused_sum,&fq,&fg,&fk,&fv,&uq,&ug,&uk,&uv})
            failures+=output->verify_guards(label);
        failures+=a.verify_guards(label)+b.verify_guards(label)+qa.verify_guards(label)+qb.verify_guards(label);
        if (fused_workspace.used()!=0 || unfused_workspace.used()!=0 ||
            fused_workspace.peak_used()!=capacity || unfused_workspace.peak_used()!=capacity) {
            std::cerr<<label<<": exact capacity/lifetime mismatch\n";++failures;
        }
        if (producer_workspace.used()!=0 || quantizer_workspace.used()!=0 ||
            producer_workspace.peak_used()!=quant_capacity || quantizer_workspace.peak_used()!=quant_capacity) {
            std::cerr<<label<<": quantizer capacity/lifetime mismatch\n";++failures;
        }
    }
    return failures;
}

template<class F> int rejects(const char* label,F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return 0; }
    std::cerr<<label<<": invalid fused call accepted\n";return 1;
}

int validation(const DevicePackedWeight& parent) {
    const TableScope table(ops::detail::LinearRouteTable::Legacy);
    constexpr int T=5;
    const auto w=parent.view();
    GuardedBf16Tensor input(D,T),res(D,T),gain(D,1),sum(D,T),qbuf(6144,T),gbuf(6144,T),kbuf(1024,T),vbuf(1024,T);
    Tensor x=input.tensor(),r=res.tensor(),g=gain.tensor(),s=sum.tensor(),q=qbuf.tensor(),gate=gbuf.tensor(),k=kbuf.tensor(),v=vbuf.tensor();
    const auto capacity=ops::attn_input_proj_workspace_capacity_bytes(w.qtype,N,D,policy,T,T);
    WorkspaceArena workspace(capacity);
    auto call=[&](const Tensor& a,const Tensor& norm,float e,Tensor& query,const Tensor* rr=nullptr,Tensor* ss=nullptr) {
        ops::attn_input_proj_fused_rmsnorm_fp8(a,norm,e,w,query,gate,k,v,policy,workspace,nullptr,rr,ss);
    };
    int failures=0;
    for (float e : {0.0F,-1.0F,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
        failures+=rejects("eps",[&]{call(x,g,e,q);});
    auto wrong=x;wrong.dtype=DType::FP32;
    failures+=rejects("input dtype",[&]{call(wrong,g,eps,q);});
    wrong=x;wrong.nb[1]+=2;
    failures+=rejects("input stride",[&]{call(wrong,g,eps,q);});
    wrong=g;wrong.ne[0]=D-1;
    failures+=rejects("norm shape",[&]{call(x,wrong,eps,q);});
    wrong=q;wrong.data=x.data;
    failures+=rejects("output overlaps input",[&]{call(x,g,eps,wrong);});
    wrong=q;wrong.data=gate.data;
    failures+=rejects("outputs overlap",[&]{call(x,g,eps,wrong);});
    wrong=s;wrong.data=r.data;
    failures+=rejects("residual sum overlaps residual",[&]{call(x,g,eps,q,&r,&wrong);});
    failures+=rejects("sum without residual",[&]{call(x,g,eps,q,nullptr,&s);});
    failures+=rejects("workspace overlaps input",[&]{
        WorkspaceArena overlap(DeviceSpan{x.data,capacity});
        ops::attn_input_proj_fused_rmsnorm_fp8(x,g,eps,w,q,gate,k,v,policy,overlap,nullptr);
    });
    if (ops::attn_input_proj_fused_rmsnorm_fp8_eligible(w,ops::LinearPolicy::A16Only,T) ||
        ops::attn_input_proj_fused_rmsnorm_fp8_eligible(w,policy,0)) {
        std::cerr<<"invalid fused eligibility\n";++failures;
    }
    auto bad_weight=w;bad_weight.n=N-1;
    failures+=rejects("projection shape",[&]{
        ops::attn_input_proj_fused_rmsnorm_fp8(x,g,eps,bad_weight,q,gate,k,v,policy,workspace,nullptr);
    });
    failures+=rejects("A16 policy",[&]{
        ops::attn_input_proj_fused_rmsnorm_fp8(x,g,eps,w,q,gate,k,v,ops::LinearPolicy::A16Only,workspace,nullptr);
    });
    return failures;
}
}

int main() {
    using namespace ninfer;
    using namespace ninfer::test;
    if (cuda_unavailable() || !ops::current_device_ada_fp8()) {
        std::cout<<"SKIP: fused RMSNorm FP8 requires native Ada FP8\n";return 77;
    }
    try {
        DevicePackedWeight parent(make_weight());
        int failures=validation(parent);
        for (int tokens : {5,17,33,65,1024}) {
            failures+=run_case(parent,tokens,false);
            failures+=run_case(parent,tokens,true,false,tokens!=17);
        }
        failures+=run_case(parent,5,true,true);
        failures+=parent.verify_preserved("fused RMSNorm FP8 weight");
        std::cout<<(failures ? "FAIL" : "OK")<<" fused RMSNorm FP8 exact pipeline and FP64 composition\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
