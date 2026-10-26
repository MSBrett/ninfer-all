# KISS proposal: put the RTX 4090 RT cores to work in NInfer

Status: revised proposal; no new kernel implementation or measured speedup.
Updated: 2026-10-25. Target: Windows, RTX 4090, Qwen3.8-27B, text-only coding agent.

## Decision

**Find one real mathematical operation in NInfer that the RTX 4090's RT hardware can
perform, implement the offload, and demonstrate that its result contributes to inference.**

The owner wants useful work on silicon that is otherwise idle. The search covers the text
inference path: layer arithmetic, reductions, comparisons, sampling, speculative decoding and
state/KV bookkeeping. MTP argmax is one candidate, not the objective or a required first choice.
A complete matrix multiplication on RT cores is not required. Geometric reformulations are
explicitly in scope when their outputs implement the selected mathematical operation.

First establish **feasibility and actual hardware attribution**. Then measure cost, overlap
and performance. A correct RT offload can establish feasibility even if its first implementation
is small or slower overall; report that result honestly rather than rejecting the experiment
before it exists. Faster complete inference is a later adoption criterion, not a prerequisite
for answering whether any real math can be offloaded.

A busy loop, discarded result, or CUDA shader doing all the arithmetic does not meet this goal.
The [original RT proposal](RT_PROPOSER_PROPOSAL.md) remains supporting research.

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

| Setting | Current working reference |
|---|---|
| Model | Qwen3.8-27B with its native MTP component |
| GPU | RTX 4090, native `sm_89` build |
| Toolchain | CUDA Toolkit 13.4.3; NVCC 13.4.92; server built successfully |
| Input | Text only; Vision disabled |
| Target KV | `rk8v4` |
| Active requests | One |
| Deployment baseline | MTP4 with the artifact's existing optimized proposal head |
| Context/KV capacity | 239,872 tokens; startup/memory fit verified, long-context execution unqualified |
| Output | 65,536 default; clients may request 131,072 when the remaining context permits |
| Thinking | `xhigh`, preserved reasoning; Qwen thinking-mode sampling |
| API / monitor | Ports `8899` / `8898` |

The deployment reached 97.0% physical VRAM usage with the Windows evictable-memory budget.
An RT experiment must account for its extra allocations and any KV-capacity tradeoff; do not
silently shrink context or present a changed configuration as a performance improvement.
The local MTP4 speed reference, about 184.8 tokens/s, was measured separately with 32K input,
100K allocated context, greedy sampling, thinking off and the optimized proposal head. It is
not a measured speed for the new thinking configuration or a full-head RT baseline.

Context includes input, reasoning and generated answer. Host prefix caching helps reuse,
but does not enlarge active GPU context. Keep model/head, KV representation, draft count and
sampling fixed for each off/on comparison. Full-head selection is a separate explicit axis
when an experiment needs it, not a prerequisite for investigating every other math operation.

## The head flags

Qwen's native MTP component and NInfer's reduced proposal matrix are separate things.
The smaller matrix is a community/NInfer converter optimization: frequency-ranked output-head
rows are copied into an additional matrix, normally 131,072 rows.

| Option | Behavior |
|---|---|
| `--lm-head-draft` | Require the artifact's optimized proposal matrix; fail if unavailable. |
| `--full-draft-head` | Use the target model's full output matrix for MTP proposals; an explicit experimental variant. |
| Neither | Automatically prefer the optimized proposal matrix when stored in the artifact. |

All choices still require `--spec mtp` to select native MTP. Target verification retains the
full vocabulary. Removing `--lm-head-draft` alone does not force the full head.

The local desktop launcher `start-local-mtp4.ps1` uses `--lm-head-draft` and four draft tokens.
Retain that working configuration as a control. A full-head experiment must select its head
explicitly and establish its own baseline; the earlier measured throughput does not transfer.

## Existing inference code to inspect

```text
Native MTP input preparation
          |
Existing MTP neural layer and KV attention
          |
Selected output-head projection and token selection
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

This source map is an example entry point, not an MTP-only boundary. Inventory actual mathematical
operations across the text path and ask whether a built-in RT intersection/traversal result can
replace some of their existing work. An operation does not have to be the largest bottleneck to
qualify for this feasibility experiment. Profiling can prioritize candidates and later explain
the result; it must not turn the question into a different CUDA/Tensor Core optimization project.

Keep the selected model, proposal head, weight/KV representations and public math semantics.
No additional vocabulary shortlist, pruning, sparse target attention or new neural drafter is
introduced merely to make an RT demonstration easier. Preserve existing state ownership.

## Research disposition

Turing's fused-computation examples may suggest ways to avoid intermediate writes and launches.
Its low-rank compression, pruning and approximate search are not selected for this deliverable.
Its simulated speculation throughput and synthetic vector retrieval are not model-speed or
coding-quality evidence. Its BSL 1.1 code is not imported.

Use the published implementations above to identify an exact offload in the existing inference
path. Keep RT research in scope; distinguish the operation already demonstrated by the authors
from any proposed NInfer adaptation. No adaptation or performance claim is established yet.

## Candidate operations

These are candidates to investigate, not implemented NInfer features. Start with an operation
that has a real consumer in the current engine and a concrete geometric mapping.

| Mathematical work | RT mapping to investigate | Starting evidence / open question |
|---|---|---|
| Minimum/maximum and argmin/argmax reductions | Encode values as geometry; a closest hit identifies an extremum. | RTXRMQ demonstrates the reduction mapping. Choose the simplest real NInfer use, not automatically MTP selection. |
| Range, threshold and membership comparisons | Encode the numeric domain as indexed geometry; intersections identify qualifying values. | RTScan and RTIndeX are leads. Identify an actual inference predicate and preserve its exact value domain. |
| Sparse-coordinate matching or index comparisons | RT traversal finds matching coordinates or keys. | RTSpMSpM supplies a matching example; its multiply/accumulate is shader work. Confirm a relevant consumer before adapting it. |
| Arithmetic reformulated as geometric intersections | Investigate whether an existing arithmetic expression can be represented by the built-in intersection result. | Open research candidate. Require an explicit derivation and actual RT execution; do not claim generic RT dot products or matrix arithmetic from shader code. |

Keep the search open to another operation if its mapping is simpler. No candidate is selected
solely because a profiler says it dominates runtime, and no unsupported arithmetic capability
is assumed because the silicon is idle.

## KISS first implementation

1. Select **one** existing mathematical operation and write down its inputs, outputs, exactness
   requirements, current CUDA/host implementation and the proposed geometric formulation.
   Prefer a demonstrated mapping such as an extremum or comparison before inventing one.
2. Build the smallest standalone offload using real values/shapes from that operation and the
   SDK's smallest applicable example. Inspect licenses before reusing a published implementation.
   Complete the implementation batch before qualification; no per-edit test campaign.
3. Show that the RT-produced value matches the operation's independent oracle and is usable
   at that existing engine boundary. Cover the represented BF16/FP32 or integer semantics,
   tie-breaking, padding and supported exceptional inputs. State unsupported domains explicitly.
4. Establish hardware attribution: which work uses built-in RT traversal/intersection and which
   work remains on SMs. Use supported tooling/counters where available and state their limits.
   An OptiX launch by itself does not establish that its arithmetic ran on RT units.
5. Route one actual operation through the offload under one opt-in switch, retaining the current
   implementation as the off path. Its result must feed the inference computation. Keep fixed
   allocation and ownership boundaries; disabled experiments must not retain their extra work.
6. Report the complete cost: encoding, index construction/update, tracing, decoding, launches,
   synchronization and persistent/temporary memory. A slower first result still answers the
   feasibility question; it does not establish a deployment improvement.
7. After the offload works, try overlap with genuinely independent CUDA/Tensor Core work where
   dependencies permit it. Measure the overlap rather than assuming separate units run for free.
   Add another operation or scheduling switch only when the first result is understood.

For changing values such as logits, charge geometry/BVH updates each step unless reuse has
actually been demonstrated. For static data, report construction separately and its amortization.
Keep RT research in scope if the first mapping fails; report why that mapping failed and select
another concrete mathematical operation. Do not replace this feasibility task with a general
bottleneck hunt or require a predicted end-to-end speedup before attempting any offload.

## Completion and measurement

The first deliverable is one table:

| Operation | RT work / SM work | Correctness | Useful inference consumer | RT execution evidence | Extra memory | Complete time / CUDA time | Overlap |
|---|---|---|---|---|---|---|---|
| One selected operation | To establish | To establish | To identify | To establish | To measure | To measure | Initially serialized |

Feasibility succeeds when real RT hardware performs part of an existing mathematical operation,
the result is correct, and that result is consumed by inference. RT activity is part of this
milestone's evidence. Do not substitute dummy utilization for useful work or claim a measured
RT-busy percentage when the tools cannot establish it.

Then evaluate engineering value separately: full-operation and inference latency, throughput,
SM/Tensor work displaced, concurrent execution, extra VRAM and usable context. Keep settings fixed,
include setup/update costs and disclose slower cases. Use the existing runner for inference
comparisons; no generic feature framework or exhaustive combinations before features exist.

A correct but slower offload is a documented feasibility result and stays experimental/off by
default. A production default requires a worthwhile measured tradeoff. Do not silently reduce KV
capacity to manufacture a gain. Stop after reporting the first decisive feasibility result before
expanding scope. This revision changes the proposal only; it installs nothing, starts no server,
and runs no compilation, GPU operation or benchmark.

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
  `.serena/cache/start-local-mtp4.ps1`, `.serena/cache/local-mtp4.requests.jsonl`,
  `profiles/bench/spec-32kin-16kout-cuda134/`,
  `profiles/bench/spec-32kin-16kout-mtp4-cuda134/`, installed NVCC output and the
  October 25 `nvidia-smi` reading of 23,828 / 24,564 MiB.
- Owner clarification, October 25: the goal is to find any real math operation that can use
  idle RT hardware; feasibility comes before deciding whether the first implementation is faster.
