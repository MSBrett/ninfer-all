// Same-table complete-stage comparison. Setup/restoration is outside both timed
// arms; the unfused arm includes residual_add when requested, RMSNorm, and A8
// attention. This is an Op-stage benchmark, not an inference-speed claim.
#include "core/device.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/linear/common/route_table.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
constexpr int D=5120,N=14336;
constexpr float eps=1.0e-6F;
constexpr auto policy=ops::LinearPolicy::AllowA8;

struct Output {
    DeviceBuffer qb,gb,kb,vb;
    Tensor q,g,k,v;
    explicit Output(int t) : qb(std::size_t(6144)*t*2),gb(qb.bytes),
        kb(std::size_t(1024)*t*2),vb(kb.bytes),q(qb.p,DType::BF16,{6144,t}),
        g(gb.p,DType::BF16,{6144,t}),k(kb.p,DType::BF16,{1024,t}),v(vb.p,DType::BF16,{1024,t}) {}
};

template<class Prepare>
bench::ColdTiming warm_prepared(Prepare&& prepare,const bench::TimedGraph& graph,
                               cudaStream_t stream,int warmup,int repeat) {
    for (int i=0;i<warmup;++i) { prepare(stream);graph.launch(stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<double> times;
    for (int i=0;i<repeat;++i) { prepare(stream);times.push_back(graph.launch_timed(stream)); }
    return bench::summarize_timings(std::move(times));
}
}

int main(int argc,char** argv) {
    using namespace ninfer;
    try {
        int warmup=5,repeat=30;
        std::string csv;
        for (int i=1;i<argc;++i) {
            const std::string flag=argv[i];
            if (flag=="--help") {
                std::cout<<"Usage: ninfer_attn_input_proj_fused_rmsnorm_fp8_bench "
                    "[--warmup N] [--repeat N] [--csv-out PATH]\n"
                    "T=5,17,65,1024; optional residual off/on; warm/cold; two reversed-order replicates.\n";
                return 0;
            }
            if (++i==argc) throw std::invalid_argument("missing benchmark option value");
            if (flag=="--warmup") warmup=std::stoi(argv[i]);
            else if (flag=="--repeat") repeat=std::stoi(argv[i]);
            else if (flag=="--csv-out") csv=argv[i];
            else throw std::invalid_argument("unknown benchmark option: "+flag);
        }
        if (warmup<0 || repeat<=0) throw std::invalid_argument("invalid benchmark counts");
        auto packed=bench::make_fp8_weight(N,D);
        DeviceBuffer flush(std::size_t{256}<<20);
        cudaStream_t stream=nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        std::ofstream file;
        if (!csv.empty()) { file.open(csv);if (!file) throw std::runtime_error("cannot open CSV"); }
        const char* header="rep,T,residual,cache,forced_table,arm,median_us,min_us,p95_us,graph_nodes,workspace_bytes,norm_buffer_bytes";
        std::cout<<"# Same A8 consumer table; restoration excluded from both arms. No model-speed claim.\n"<<header<<'\n';
        if (file) file<<header<<'\n';
        for (int tokens : {5,17,65,1024}) {
            const auto table=tokens==5 ? ops::detail::LinearRouteTable::Legacy : ops::detail::LinearRouteTable::Unified;
            ops::detail::force_linear_route_table(table);
            if (!ops::attn_input_proj_fused_rmsnorm_fp8_eligible(packed.weight,policy,tokens))
                throw std::runtime_error("requested same-table fused RMSNorm FP8 route is not eligible");
            DeviceBuffer xb=bench::make_bf16(std::size_t(D)*tokens,101U);
            DeviceBuffer rb=bench::make_bf16(std::size_t(D)*tokens,103U);
            DeviceBuffer wb=bench::make_bf16(D,107U),nb(xb.bytes),sb(xb.bytes),fsb(xb.bytes);
            Tensor x(xb.p,DType::BF16,{D,tokens}),r(rb.p,DType::BF16,{D,tokens}),
                w(wb.p,DType::BF16,{D}),norm(nb.p,DType::BF16,{D,tokens}),
                sum(sb.p,DType::BF16,{D,tokens}),fsum(fsb.p,DType::BF16,{D,tokens});
            Output ordinary(tokens),fused(tokens);
            const auto capacity=ops::attn_input_proj_workspace_capacity_bytes(packed.weight.qtype,N,D,policy,tokens,tokens);
            WorkspaceArena uw(capacity),fw(capacity);
            for (bool residual : {false,true}) {
                const auto prepare=[&](cudaStream_t s) {
                    if (residual) {
                        CUDA_CHECK(cudaMemcpyAsync(sb.p,rb.p,rb.bytes,cudaMemcpyDeviceToDevice,s));
                        CUDA_CHECK(cudaMemcpyAsync(fsb.p,rb.p,rb.bytes,cudaMemcpyDeviceToDevice,s));
                    }
                };
                bench::TimedGraph ug,fg;
                ug.capture(stream,[&](cudaStream_t s) {
                    if (residual) ops::residual_add(x,sum,s);
                    ops::rmsnorm(residual ? sum : x,w,eps,true,norm,s);
                    ops::attn_input_proj(norm,packed.weight,ordinary.q,ordinary.g,ordinary.k,ordinary.v,policy,uw,s);
                });
                fg.capture(stream,[&](cudaStream_t s) {
                    ops::attn_input_proj_fused_rmsnorm_fp8(x,w,eps,packed.weight,
                        fused.q,fused.g,fused.k,fused.v,policy,fw,s,residual ? &r : nullptr,residual ? &fsum : nullptr);
                });
                for (bool cold : {false,true}) for (int rep=0;rep<2;++rep) for (int order=0;order<2;++order) {
                    const bool fused_arm=(rep+order)%2!=0;
                    const auto& graph=fused_arm ? fg : ug;
                    const auto timing=cold
                        ? bench::measure_cold_graph_prepared(prepare,graph,flush,stream,warmup,repeat)
                        : warm_prepared(prepare,graph,stream,warmup,repeat);
                    const auto write=[&](std::ostream& out) {
                        out<<rep<<','<<tokens<<','<<residual<<','<<(cold ? "cold" : "warm")<<','
                            <<(table==ops::detail::LinearRouteTable::Legacy ? "legacy" : "unified")<<','
                            <<(fused_arm ? "fused" : "unfused")<<','<<timing.median_us<<','<<timing.min_us<<','
                            <<timing.p95_us<<','<<graph.nodes()<<','<<capacity<<','<<(fused_arm ? 0 : nb.bytes)<<'\n';
                    };
                    write(std::cout);if (file) { write(file);file.flush(); }
                }
            }
            ops::detail::force_linear_route_table(std::nullopt);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        ninfer::ops::detail::force_linear_route_table(std::nullopt);
        std::cerr<<"fused RMSNorm FP8 benchmark: "<<e.what()<<'\n';return 1;
    }
}
