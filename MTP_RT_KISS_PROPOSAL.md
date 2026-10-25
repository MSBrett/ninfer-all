# KISS proposal: native MTP and useful RT offloads on the RTX 4090

Status: revised proposal; no new kernel implementation or measured speedup.
Updated: 2026-10-25. Target: Windows, RTX 4090, Qwen3.8-27B, text-only coding agent.

## Decision

**Use Qwen's native MTP, preserve its full vocabulary and verification, and use idle RT
hardware for useful work wherever published implementations provide a relevant starting point.**

The owner accepts a small useful offload anywhere in inference, including selection, indexing
or data access. A complete MTP matrix multiplication on RT cores is not required.
The earlier exclusion of geometric mappings was too broad and is withdrawn. The earlier
approximate vocabulary shortlist remains outside the selected baseline.

The [original RT proposal](RT_PROPOSER_PROPOSAL.md) remains supporting research.
This file is the focused proposal; the table below records prior art, not NInfer speedup claims.

## Published implementations to study

RT traversal and built-in intersections use RT hardware. OptiX custom device programs execute
on SMs. Distinguish those responsibilities when attributing an application's arithmetic.

| Work | Implemented operation and hardware division | Evidence and limits |
|---|---|---|
| [RTXRMQ](https://arxiv.org/abs/2306.03282), [code](https://github.com/temporal-hpc/rtxrmq) | Range minimum: array values become triangles; closest ray hit selects a minimum. | Published RT reduction. Beats the evaluated GPU baseline for small ranges; loses for medium/large ranges. This is not a demonstration of per-token dynamic-logit argmax. |
| [RTScan](https://kay21s.github.io/RTScan2024.pdf), [code](https://github.com/AntaresAlice/RTScan) | Exact database filtering, including conjunctive predicates; RT traversal/intersections plus filter-vector processing. | Implemented on RTX 3090. Data updates require index work; authors restrict the method to infrequent updates. Scan here means predicate filtering, not a prefix sum or memory copy. |
| [RTIndeX](https://arxiv.org/abs/2303.01139) | Point/range key lookups encoded as rays through indexed geometry. | Prior art for offloading index traversal and comparisons. Not an arbitrary GPU gather/store or bulk-transfer engine. |
| [RTSpMSpM](https://github.com/escalab/RTSpMSpM) | Sparse matrix multiplication: RT finds matching sparse coordinates; the any-hit CUDA program multiplies values and atomically accumulates results. | Public implementation, not RT multiply-accumulate instructions. Dense Qwen MTP is a different workload. |
| [JUNO](https://arxiv.org/abs/2312.01712) | Product-quantized approximate vector search; pipelines RT search work with Tensor Core accumulation. | Evaluated on RTX 4090 among other GPUs. Direct prior art for using both unit types concurrently; its ANN approximation is not our full-head baseline. |
| [RT-HiSS](https://arxiv.org/html/2609.01975v1), [code](https://github.com/Gowanlock-Lab/rt-hiss) | Exact high-dimensional radius search: RT indexes three dimensions; a separate CUDA kernel computes full-dimensional distances and refines results. | Existing exact search-and-refine implementation. This does not demonstrate dense dot products on RT units or an NInfer inference improvement. |
| [GPU-RMQ](https://arxiv.org/html/2604.01811v1), [code](https://github.com/lakreis/GPU-RMQ) | Hybrid reduction with an RT top hierarchy and CUDA processing below it. | Negative result on RTX 4090: OptiX overhead and loss of warp operations outweighed RT acceleration. The authors recommend their CUDA-only variant. |

RTSpMSpM's inspected arithmetic is in
[the any-hit program](https://github.com/escalab/RTSpMSpM/blob/216add0bf257e9e586565b0a109690481bddb10f/optixSpMSpM/src/optixSpMSpM/optixSpMSpM.cu).
The multiplication and atomicAdd there are shader work; they must not be reported as RT ALU work.

These are published implementations, not speculative encodings of arbitrary arithmetic.
No cited implementation establishes a native Qwen MTP speedup, a replacement for its dense
projection, or general-purpose RT memcpy. Index construction, extra storage and SM-side work
remain part of each implementation's cost. Preserve KV capacity when evaluating adoption.

## Configuration and goal

| Setting | Choice |
|---|---|
| Model | Qwen3.8-27B with its native MTP component |
| GPU | RTX 4090, native `sm_89` build |
| Toolchain | Installed CUDA Toolkit 13.4.3; NVCC reports 13.4.92 |
| Input | Text only; Vision disabled |
| Target KV | `rk8v4` |
| Active requests | One |
| Baseline | `--spec mtp --full-draft-head`, initially three draft tokens |
| Context | Maximum demonstrated usable capacity; initially target **160,000–200,000 tokens** |
| API | Port `8899` |

The separate three-mode experiment uses 100,000 tokens. This document does not alter that work.
The deployment goal remains maximum useful context. The 160–200K range is a hypothesis;
increase further if the complete configuration fits and executes correctly. The documented
native window is 262,144 tokens; extending beyond it is a separate decision.

Context includes input and generated output. Reserve an explicit output allowance.
MTP weights and state, workspace, CUDA Graphs and desktop usage count against the same memory.
Host prefix caching helps reuse, but does not enlarge active GPU context.

The local `cuda134-sm89` CMake configuration selects Release, architecture 89 and the installed
13.4.3 toolkit. Installation/configuration alone does not establish a completed build or
qualified performance.

## The head flags

Qwen's native MTP component and NInfer's reduced proposal matrix are separate things.
The smaller matrix is a community/NInfer converter optimization: frequency-ranked output-head
rows are copied into an additional matrix, normally 131,072 rows.

| Option | Behavior |
|---|---|
| `--lm-head-draft` | Require the artifact's optimized proposal matrix; fail if unavailable. |
| `--full-draft-head` | Use the target model's full output matrix for MTP proposals; our baseline. |
| Neither | Automatically prefer the optimized proposal matrix when stored in the artifact. |

All choices still require `--spec mtp` to select native MTP. Target verification retains the
full vocabulary. Removing `--lm-head-draft` alone does not force the full head.

The inspected local `run-experiment-cuda134.ps1` MTP branch uses `--lm-head-draft`.
It needs an explicit full-head selection for this proposal's baseline. The launcher has not
been changed by this document update.

## Existing code to optimize

```text
Native MTP input preparation
          |
Existing MTP neural layer and KV attention
          |
Full vocabulary projection and token selection
          |
Existing target verification and state commit
```

| Responsibility | Source |
|---|---|
| MTP round and dependent draft steps | [program/speculative/mtp.cpp](src/models/qwen3_5/program/speculative/mtp.cpp) |
| MTP stem, attention and proposal_argmax | [execution/text.cpp](src/models/qwen3_5/execution/text.cpp) |
| Attention projection helpers | [execution/mtp.cpp](src/models/qwen3_5/execution/mtp.cpp) |
| Shared embedding and full output head | [load/mtp.cpp](src/models/qwen3_5/load/mtp.cpp) |
| Weight-format-specific matrix operations | [linear.cpp](src/ops/linear/linear.cpp) |
| Token selection | [argmax.cu](src/ops/launcher/argmax.cu) |
| RTX 4090 route choices | [device_profiles.json](src/runtime/engine/device_profiles.json) |

Start with the actual bottleneck: vocabulary scoring, attention, FFN, dequantization, copies
or launch overhead. Candidate changes are better CUDA/Tensor Core kernels, tiling, fusion and
data reuse. Do not assume a new kernel is necessary or that an existing kernel leaves useful
arithmetic capacity idle.

Keep the full vocabulary and selected weight/KV representations. No additional frequency
shortlist, low-rank truncation, activation pruning, sparse target attention or new neural drafter.
Preserve the current MTP proposal/verification semantics and model state ownership.

## Research disposition

Turing's fused-computation examples may suggest ways to avoid intermediate writes and launches.
Its low-rank compression, pruning and approximate search are not selected for this deliverable.
Its simulated speculation throughput and synthetic vector retrieval are not model-speed or
coding-quality evidence. Its BSL 1.1 code is not imported.

Use the published implementations above to identify an exact offload in the existing inference
path. Keep RT research in scope; distinguish the operation already demonstrated by the authors
from any proposed NInfer adaptation. No adaptation or performance claim is established yet.

## Independent switches and combinations

Every implemented offload gets one independent on/off switch in the development runner.
All off selects the existing CUDA path. Off must also remove that feature's index, workspace
and update work, so the baseline does not pay for disabled features. Log the effective settings;
fail clearly if an enabled feature is unsupported rather than silently running the baseline.
These are proposed experimental controls, not existing CLI options.

| Control | Initial scope |
|---|---|
| RT selection | First prototype: full-vocabulary MTP token selection, with the existing CUDA selector as baseline. |
| RT overlap | Separate scheduling control, introduced only when there is independent work to overlap; serialized execution remains measurable. |
| Other offloads | Give each a separate switch when an actual NInfer operation and implementation are selected. Predicate filtering, index lookup and sparse matching remain prior-art leads, not placeholder features. |

Feature switches belong to operations, not to paper names. Reuse the existing runner and
explicit configuration; do not build a generic dispatch framework or implement every paper.
Keep native MTP, full-head choice, model, KV format, context, draft count and sampling fixed
while comparing RT combinations. Existing MTP enable/disable is a separate experimental axis.

For the small number of implemented features, compare all off, each alone, relevant pairs and
all together. Compare serialized and overlapped scheduling for combinations with independent
work. If only two features exist, measure all four on/off combinations. Do not infer their
combined benefit by adding isolated speedups. Never overlap a consumer with its unfinished
producer: the next dependent MTP draft cannot start before its required token is selected.

## KISS first implementation

The first concrete candidate is an RTXRMQ-style selector at the existing proposal_argmax seam.
It has a small integration boundary and leaves the neural layer, full head projection, KV and
verification intact. This is a choice for minimum prototype effort, not a claim that dynamic
logit selection will beat CUDA or be the largest opportunity.

1. Build a small replay prototype using actual full-head MTP logits and token counts, the
   existing CUDA selector, and one OptiX selection path. Use the SDK's smallest applicable
   sample and published RTXRMQ mapping as references; inspect reuse licenses before copying.
   Do not start with server integration, multiple RT offloads or concurrent scheduling.
2. Include geometry encoding, per-step BVH build/update, tracing, decoding, launches and
   synchronization in the RT cost. Static-scene trace-only timings cannot answer this question:
   logits change every draft step. Report persistent and temporary memory separately.
3. Complete this implementation batch before qualification. Independently establish the winner
   for the represented logit type; match token IDs, tie-breaking, padding exclusion and the
   supported handling of non-finite values. An encoding that cannot preserve those semantics
   is not an exact replacement. No vocabulary shortlist or approximate winner is permitted.
4. Measure the full selector cost against the existing CUDA selector at actual decode batch
   sizes, not a large synthetic query batch. Record the baseline selector's share of an MTP
   round to bound its possible end-to-end payoff. If full cost loses, stop this prototype before
   adding it to serving. That rejects this offload, not every use of RT cores.
5. If it wins, add the smallest opt-in integration at proposal_argmax and compare complete MTP
   rounds and output throughput with the switch off/on. Preserve Model/Program/Op ownership,
   fixed allocations and state semantics; expose graph/interop costs in the measurement.
6. Only after that result, select a second concrete offload from the prior art and give it its
   own switch. Add overlap only where dependencies permit it. Hill-climb one parameter at a
   time from the best demonstrated configuration: layout, update strategy, batching, then
   scheduling as applicable. Retain the original baseline for every comparison.

If the initial timing bound shows selection cannot yield a useful gain, record that result
and choose another concrete operation before doing a larger selector implementation.
A useful outcome can be small, but must reduce a measured cost rather than just raise RT usage.

## Measurement and stopping rule

Keep workload, context, output allowance and all non-feature settings identical across a
comparison. Include representative code generation, editing and cached tool turns once the
prototype reaches inference. Use independent arithmetic/exact oracles for changed operations.
Batch development before qualification; no per-edit test runs or unrelated suite campaigns.

Record a compact comparison table with enabled features, scheduling, component time, complete
MTP round time, output tokens/s, acceptance, cold/warm latency, extra VRAM and correctness.
Repeat enough to distinguish a claimed difference from observed run variation. Include slower
cases and setup/update costs. Kernel speed alone does not establish an inference improvement.
RT utilization is diagnostic, not the success criterion.

Keep context fixed for performance comparisons. Then separately establish maximum usable
context with the selected configuration and report any capacity lost to RT indices/workspace.
Do not silently lower context to make an offload fit. The 160–200K goal remains unverified.

Retain only changes with a demonstrated end-to-end benefit and acceptable correctness and
memory use; keep optional experiments off by default until qualified. Once the first result
is decisive, report it before expanding the implementation. No installation, compilation or
GPU execution has been performed for this document update. Move supported behavior/results
into the existing active references and retire this temporary proposal when the work ends.

## Sources

Inspected/reviewed 2026-10-25:

- [NVIDIA Ada architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf):
  dedicated RT functions and the GPU's separate execution units.
- [NVIDIA OptiX explanation](https://forums.developer.nvidia.com/t/program-rt-cores-using-opengl-vulkan-or-cuda/247089/6):
  supplied device code executes on SMs; RT acceleration covers traversal/intersection.
- [Original RT research](RT_PROPOSER_PROPOSAL.md), [CLI](docs/cli.md#speculative-decoding),
  [converter](tools/convert/proposal.py), [head selection](src/runtime/engine/model_instance.cpp),
  [Engine ownership](docs/maintainer/engine-architecture.md) and the source map above.
- [Turing source](https://github.com/intutic/turing/tree/82f039c10981b450faf2abae2e9b60383dca0ec2):
  read-only inspection, no reproduced results or imported code.
- Local `.serena/cache/cuda134-sm89/CMakeCache.txt`,
  `.serena/cache/run-experiment-cuda134.ps1`, and installed NVCC version output.
