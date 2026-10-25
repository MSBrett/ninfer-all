# Proposal: RT acceleration across the NInfer pipeline

Status: research proposal; no implementation or measured NInfer speedup.
Research and last update: 2026-10-25. Initial target: Windows, RTX 4090, Qwen3.8-27B.
Source inspection baseline: `b3402a6fd6b69cafad394d2213721be16cb2a90d`, including the local checkout's existing changes.

## Decision proposed

Investigate [[RT cores]] as search and pruning hardware across input encoding, prefill, decode,
drafting and data placement. The first bounded experiment remains vocabulary candidate search
for the existing [[speculative decoding]] proposer. Keep the MTP/DFlash backbone and target
verifier. Start with DFlash2, whose proposal positions can be searched together; evaluate MTP
separately because its draft steps are dependent. This is the smallest integration boundary,
not a claim that drafting is the only useful application of the hardware.

The hypothesis is technically plausible, not an established performance result. Published work
supports dynamic vocabulary selection and RT-assisted approximate search separately. The arXiv
search below found no direct demonstration combining RT cores with an MTP or DFlash proposer.
That search result is not a novelty claim or proof that no such work exists.

This document proposes a bounded feasibility study. It does not authorize installations,
changes to the active Windows build, or immediate GPU runs. Batch implementation before
verification; schedule any experiment separately from the ongoing build and qualification.

## Pipeline opportunity map

The useful hardware operation is geometric traversal and candidate rejection. The opportunity
is often to avoid reading or calculating irrelevant data. Ray-generation and custom intersection
programs still execute programmable GPU instructions; putting work inside OptiX does not make
all of its arithmetic run on RT cores. There is no supported RT replacement for arbitrary DMA,
matrix multiplication, quantized-weight decoding, or GDN recurrence.

The entries below are NInfer research hypotheses, not existing capabilities or measured gains.

| Stage | Candidate RT operation | Semantic and engineering boundary | Priority |
|---|---|---|---|
| Input / vision encoding | Find similar regions, patches, frames or cached inputs. | Similarity can nominate reuse candidates; exact reuse needs the complete input/position/context identity. Vision attention makes whole-encoder feature reuse more demanding than patch similarity. Frame/token dropping changes model inputs. | Exploratory. |
| Prefill | Search key/block candidates for many attention queries. | More queries may amortize launch cost, but dense tiled attention is a strong baseline. Omitting contributions approximates target mathematics; include index construction in prefill latency. | Separate attention study. |
| Long-context decode | Search a KV index, refine candidate blocks, and request their pages. | Potentially saves attention work and memory traffic. Applies to full-attention layers, not the GDN recurrence. Dynamic state/index costs and attention quality are central. | Strongest broader hypothesis. |
| Target output head | Retrieve candidate vocabulary rows before target selection. | Exact greedy top-1 needs conservative exclusion bounds and stable ties. Sampling needs its complete processed distribution or a separately proven exact sampling procedure. A good shortlist alone is insufficient. | After draft-head study. |
| MTP / DFlash proposal head | Search immutable vocabulary weights and rerank a bounded set. | Existing target verification permits approximate proposals, provided actual proposal semantics are retained. | First experiment. |
| Retrieval proposals | Match target hidden state against stored contexts and propose their continuations. | Additional datastore, model-specific alignment, memory and update lifecycle. | Secondary proposer study. |
| MoE prefetch | Predict which expert banks will be needed and initiate transfers. | Incorrect predictions may waste bandwidth but cannot replace authoritative routing or omit required experts. The initial dense 27B target has no routed experts. | Relevant to other supported families. |
| Cache / load-store planning | Batch point/range/overlap queries over page or artifact metadata. | RT can produce IDs; existing I/O and copy engines move bytes. Compare CPU interval/hash structures and CUDA first, especially with only one to eight requests. | Low until a large irregular metadata bottleneck is measured. |
| Conversion / codebooks | Search large vector-quantization codebooks. | Selection must satisfy the codec's contract. Small scalar tables are poor candidates; large codebooks are not an assumed current feature. | Exploratory. |

The current Windows DirectStorage change concerns file reads into host memory for page restore.
It does not establish a GPU-direct KV search/fetch path or measured inference acceleration.
An RT-selected page list would be a separate integration with existing residency, ownership and
transfer contracts; do not relabel ordinary copies as RT work.

Research order: draft vocabulary search, then long-context KV selection, then exact target-head
search. Media reuse, expert prefetch and metadata indexing remain scoped hypotheses requiring
their own relevant workloads. GPU utilization or silicon area alone is not an acceptance metric.

## Current engine and opportunity

The [DFlash reference](docs/maintainer/dflash.md) defines the current model semantics. The
27B DFlash2 has hidden width 5120, five draft layers, and a rank-256 conditional selector over
16 candidates per proposal position. The usual seven proposed tokens follow one anchor in
an eight-position block. Supported startup draft counts are 1 through 15.

After the backbone, [draft.cpp](src/models/qwen3_5/execution/draft.cpp) normalizes the mask
hidden states, calls `linear_topk`, projects selector features, and calls
`candidate_selector_path`. The full or optimized proposal head is selected at startup.
The optimized head already restricts the 27B vocabulary to 131,072 of 248,320 rows;
the [CLI reference](docs/cli.md) documents that representation and its memory cost.
This is the relevant optimized baseline, not an unoptimized full-vocabulary framework call.

For hidden state `h` and vocabulary weight row `w_v`, head selection seeks high scores
`s_v = dot(w_v, h)`: [[maximum inner-product search]]. A search index might avoid scanning
most weight rows. RT cores would accelerate geometric candidate discovery; CUDA/Tensor Cores
would still perform projection, score refinement, the neural backbone, and target verification.

No 4090 measurement currently establishes that head scoring dominates our drafting or complete
round time. Existing documentation's measurements on another card are not a substitute.

## Prior art and evidence limits

All links below were accessed on 2026-10-25. Reported results belong to their authors and
workloads; none is a projected NInfer speedup. Versioned links identify the papers reviewed,
not dependency pins for a future implementation.

| Work | Relevant result or method | What it does not establish |
|---|---|---|
| [JUNO, arXiv:2312.01712v1](https://arxiv.org/abs/2312.01712v1); [ASPLOS 2024 paper](https://horizon-lab.org/pubs/asplos24-juno.pdf) | Product-quantized approximate vector search mapped to RT hardware, with CUDA/Tensor work. The conference paper describes inner-product support and evaluates an RTX 4090. | Low-latency vocabulary search for a few 5120-dimensional hidden states. Its search-throughput gains cannot be transferred to token generation. |
| [SpecVocab, arXiv:2602.13836v2](https://arxiv.org/abs/2602.13836v2); [code](https://github.com/SamsungLabs/SpecVocab) | Predicts a context-dependent vocabulary subset, then scores original output rows. Reports up to 8.1% higher average throughput over EAGLE-3. | RT acceleration or gains on our quantized DFlash2 head. Its learned predictor is a competing CUDA approach, not a component we can assume is already trained. |
| [FR-Spec, arXiv:2502.14856v2](https://arxiv.org/abs/2502.14856v2) | Frequency-ranked vocabulary restriction reduces draft-head work; reports 1.12x average speedup over EAGLE-2 in its experiments. | Additional benefit beyond our existing static proposal vocabulary, or a reason to compare only against the full head. |
| [DFlash, arXiv:2602.06036v2](https://arxiv.org/abs/2602.06036v2); [official code](https://github.com/z-lab/dflash) | Generates a block in one draft forward using target features. Supplies the draft/verify latency-versus-acceptance framework. | An RT implementation. The original paper is not the complete DFlash2 selector specification. |
| [DFlash2 implementation in SGLang](https://github.com/sgl-project/sglang/pull/35371) | Implements dynamic convolution, top-k conditional selection, and retention of the realized proposal distribution for verification. | Portability of its H200 results to our 4090 or a benefit from RT hardware. |
| [DReSD, arXiv:2502.15572v2](https://arxiv.org/abs/2502.15572v2) | Retrieves continuations with contextual hidden states; reports 19% faster generation on average than REST. Dense stores have substantial storage and preprocessing costs. | Acceleration of the existing neural head. This is an alternative proposer with an additional datastore. |
| [REST, arXiv:2311.08252](https://arxiv.org/abs/2311.08252) | Establishes retrieval of token continuations as a speculative drafting method. | A need for RT hardware; exact token lookup has simpler implementations. |
| [RT-HiSS, arXiv:2609.01975v1](https://arxiv.org/abs/2609.01975v1) | RT pruning followed by CUDA refinement for high-dimensional fixed-radius Euclidean searches. | A ready-made top-k inner-product search. Its exactness applies to its own search problem. |
| [Fast Inference from Transformers via Speculative Decoding, arXiv:2211.17192v2](https://arxiv.org/abs/2211.17192v2) | Establishes distribution-preserving speculative verification with approximate proposals. | That arbitrary proposal changes remain correct without supplying the distribution actually sampled. |

The exploratory [SpectralAI project](https://github.com/JordiSilvestre/Spectral-AI) is related
RT/MoE-routing prior art, not a validated baseline for this dense model. Its README reports a
CUDA routing variant faster than its RT variant, routing as only 2.8% of inference time in its
profile, and quality losses in some configurations. These limitations prevent treating its
large routing-only ratios as evidence of an end-to-end win here.

Searches included `site:arxiv.org "ray tracing" "speculative decoding"`,
`site:arxiv.org "RT cores" "speculative"`, `site:arxiv.org "ray tracing cores" "draft"`,
and searches for vocabulary speculation, FR-Spec, DFlash, and retrieval-based drafting.
These are bounded literature searches, not an exhaustive systematic review.

## GitHub implementation review

Reviewed read-only on 2026-10-25; no repository was cloned, built or benchmarked for this review.
Source links below identify inspected files on their default branches. Conclusions describe
those files, not an independent reproduction of authors' performance claims.

| Implementation and inspected source | Transferable lesson | Limit for NInfer |
|---|---|---|
| [RTNN](https://github.com/horizon-research/rtnn), [geometry.cu](https://github.com/horizon-research/rtnn/blob/main/src/optixNSearch/geometry.cu), [search.cpp](https://github.com/horizon-research/rtnn/blob/main/src/optixNSearch/search.cpp) | Uses RT bounding-box traversal, with custom distance tests and top-k maintenance. Query ordering/batching and bounded result storage are explicit parts of the design. | Operates on low-dimensional points. Sphere-distance arithmetic and top-k code are programmable work, not proof that RT hardware computes arbitrary dot products. |
| [Arkade](https://github.com/MDurgaKeerthi/Arkade), [deviceCode.cu](https://github.com/MDurgaKeerthi/Arkade/blob/main/src/s01-knn/deviceCode.cu), [hostCode.cpp](https://github.com/MDurgaKeerthi/Arkade/blob/main/src/s01-knn/hostCode.cpp); [paper](https://arxiv.org/abs/2311.09168) | Filter/refine and metric transformations make non-Euclidean searches possible. The custom intersection program calculates the configured metric. | The inspected sample uses x/y/z coordinates, with host input z set to zero. Its dot mode is not a 5120-dimensional MIPS implementation or justification for discarding row norms. |
| [TrueKNN](https://github.com/vani-nag/OWLRayTracing); [paper](https://arxiv.org/abs/2305.18356) | The paper grows the radius until enough neighbors exist, avoiding an arbitrary fixed-radius completeness assumption. | Repeated passes may be unacceptable for decode latency. Our bounded search needs explicit underfill behavior, not an assumed always-full result. Repository overview and paper reviewed; its implementation was not qualified. |
| [RT-HiSS](https://github.com/Gowanlock-Lab/rt-hiss), [deviceCode.cu](https://github.com/Gowanlock-Lab/rt-hiss/blob/main/deviceCode.cu), [hostCode.cpp](https://github.com/Gowanlock-Lab/rt-hiss/blob/main/hostCode.cpp), [utility.cu](https://github.com/Gowanlock-Lab/rt-hiss/blob/main/utility.cu) | Geometric filtering generates candidate groups; CUDA performs high-dimensional refinement. Grouped data layouts can improve reuse. | The inspected host path includes count/collection passes, host transfers and synchronization; refinement also allocates work buffers. Adapt the idea, not the batch analysis driver, to an allocation-free decode loop. |
| [NVIDIA cuBQL](https://github.com/NVIDIA/cuBQL), [shrinkingRadiusQuery.h](https://github.com/NVIDIA/cuBQL/blob/main/cuBQL/traversal/shrinkingRadiusQuery.h) | Flexible BVH traversal callbacks offer a way to evaluate the geometric pruning algorithm independently. | This is CUDA BVH traversal, not evidence of RT-core execution. Useful as an algorithm baseline; do not label a cuBQL gain an RT gain. |
| [RTXRMQ](https://github.com/temporal-hpc/rtxrmq); [paper](https://arxiv.org/abs/2306.03282) | Demonstrates that range extrema can map to triangle intersections, expanding the space beyond nearest neighbors. | Geometry derived from changing logits would need updates. The repository's batched range-query example is not evidence of a faster per-step vocabulary reduction. README and paper reviewed. |
| [RTSpatial](https://github.com/RTSpatial/RTSpatial) | Mutable point/range indexes suggest applications to changing page/region metadata. | The README lists high-update-load optimization as unfinished. Establish update cost and a meaningful query batch before considering cache integration. README reviewed. |
| [SpectralAI](https://github.com/JordiSilvestre/Spectral-AI), [bvh_router_kernel.cu](https://github.com/JordiSilvestre/Spectral-AI/blob/main/cuda/v5/bvh_router_kernel.cu), [optix_router_host.cpp](https://github.com/JordiSilvestre/Spectral-AI/blob/main/cuda/optix_router_host.cpp), [optix_router_hitgroup.cu](https://github.com/JordiSilvestre/Spectral-AI/blob/main/cuda/optix_router_hitgroup.cu) | Contains separate CUDA and OptiX implementations, AABB and triangle geometry, and synchronous/asynchronous launch paths. Compare primitive choices and account for synchronization. | The standalone CUDA router is not RT execution. The async host path avoids its explicit per-call device synchronization, but still uploads launch parameters and requires caller-managed dependencies; graph suitability and NInfer latency remain unverified. |

For source identity, the GitHub connector returned content-blob IDs for the inspected
RT-HiSS device code (`cba09d8cf768fb452a48f075f93bfe8c7d5f85b1`), SpectralAI standalone CUDA
router (`c1506e793506262bab985f5179eab24acd2f0c9a`) and cuBQL traversal header
(`05948d4e34ef197e1fccf8a7d3baf9082ab74651`). These identify file contents, not repository commits.

### Competing mappings and implementation constraints

[*Raster is Faster*, CIDR 2026](https://www.vldb.org/cidrdb/papers/2026/p18-doraiswamy.pdf)
compares rasterization-based indexing with RTScan on an RTX 4090. The authors report much faster
index construction and query execution for their rasterization approach. Their experiment uses
large low-dimensional database indexes and excludes BinDex filtering; it does not establish a
vocabulary-search result. It does establish a relevant competing hardware mapping. If the
candidate operation becomes a range/coverage query, evaluate a rasterization approach when the
extra graphics API/interoperability work can plausibly pay, rather than making RT usage the goal.

[SpecVocab's repository](https://github.com/SamsungLabs/SpecVocab) describes fused candidate-row
scoring and a learned vocabulary predictor. It benchmarks on H100, not our 4090. Its
[LICENSE](https://github.com/SamsungLabs/SpecVocab/blob/main/LICENSE) is CC BY-NC-SA 4.0;
do not assume the paper's linked implementation can be copied under permissive SGLang terms.
Use it as attributed prior art; resolve any source-reuse terms before importing code.

Repository availability does not imply current Windows/CUDA/OptiX compatibility. In particular,
RTNN includes old SDK headers and RT-HiSS documents older tested SDKs and Quadro cards. Learn
their mappings while selecting and qualifying the current SDK independently.

### Questions that can reject the vocabulary design

1. Do the head's important rows have geometry that supports strong pruning? A handful of
   projected dimensions can miss a winning row whose remaining components dominate the dot
   product. Nested 3D acceleration structures do not themselves recover omitted dimensions.
2. Can we retain useful candidates at a bounded budget? Measure winner/top-16 recovery and
   proposal acceptance, including near ties, rare tokens and multilingual text. Normalizing rows
   changes inner-product ranking unless the transformation explicitly preserves their norms.
3. Is refinement cheaper than today's contiguous, quantized head scan? Sparse row gathers can
   lose coalescing and weight reuse across the seven DFlash2 positions.
4. Are enough physical rays available to hide traversal latency? Subspace/PQ decomposition can
   create many rays from a few semantic queries, but its launches, result merging and index
   traffic count toward cost. Seven token positions need not mean only seven rays.
5. Does the mechanism improve the full round after actual-q storage, verification and index
   memory are included? If not, retain a useful CUDA-only result or close the experiment.

## A proposer without a neural draft model

An RT-native proposer could replace the MTP/DFlash drafting computation rather than accelerate
its head. "RT-native" means geometric lookup/selection supplies the proposals; it does not mean
the complete speculative decoder runs exclusively on RT hardware. The target verifier still
executes the model, and query preparation, ray-generation/hit programs, result selection and
token gathers need programmable GPU work. RT units are not standalone programmable processors.

### Retrieval variant: first candidate

```text
Committed target features + unprocessed anchor
                     |
Small deterministic query preparation (no neural draft forward)
                     |
RT search over stored context/continuation keys
                     |
Select one stored continuation and gather a bounded token block
                     |
Existing full target verification, correction and state commit
                     |
Retain useful committed examples for later lookup, under a bounded policy
```

Each index entry couples a model-specific contextual key and anchor with a known continuation.
Record the same processed frontier/anchor convention used by the live query; do not shift the
continuation by one token. Target-generated examples may align better than arbitrary corpus
continuations, but that is a hypothesis to measure. Target features are already available from
verification; any extra projection, normalization or feature capture still counts as proposer
cost. A learned projection would add training and arithmetic rather than being free RT work.

The first version should choose a single retrieved chain deterministically. Its proposals are
point masses and can use the existing coupled verification contract used by MTP/DFlash.
No original DFlash2 `q` can be reused after replacing its proposal mechanism. A later stochastic
retriever must define and retain its own actual conditional probabilities. A miss proposes no
extra tokens and uses the normal target step; do not run an unnecessary full-width verification
block solely to preserve the appearance of speculation.

This avoids neural draft layers, their context preparation and a vocabulary scan. It may release
their memory if they are not loaded, but the retrieval index can consume more memory than it
saves. It cannot assume useful repeated continuations exist. Cold-start, novel prose, rare
languages, domain shift and self-reinforcing low-quality retrieval need explicit coverage.

[REST](https://arxiv.org/abs/2311.08252), [DReSD](https://arxiv.org/abs/2502.15572v2) and
[SAM-Decoding](https://arxiv.org/abs/2411.10666v3) establish neural-drafter-free retrieval designs;
none of those papers establishes RT acceleration. SAM-Decoding's efficient suffix matching is
a strong comparison for copy-heavy work. [Trajectory-Retrieval Speculative Decoding](https://arxiv.org/abs/2610.07350v1),
submitted 2026-10-05, explores current-trajectory reuse and adapts retrieval from recent
verification outcomes; it combines retrieval with model drafts, rather than demonstrating a
standalone RT proposer. The bounded search on 2026-10-25 found no directly demonstrated pure
RT-based speculative proposer.

### Learned geometry variant: higher-risk extension

Distill target-generated examples into a collection of geometric regions whose leaf metadata
contains candidate next-token blocks or continuation IDs. Ray queries select regions, and
programmable code resolves overlaps or votes. This would be a new learned geometric drafter,
not an exact transcription of the existing MTP/DFlash weights.

The representation must demonstrate useful conditioning beyond a tiny low-dimensional summary.
Multiple projections/subspaces can improve capacity but add queries, merge work and memory.
Overlapping boxes do not automatically implement an arbitrary decision forest or reproduce
high-dimensional neural arithmetic. No qualifying implementation or acceptance result has been
found for this design; keep it behind the simpler retrieval experiment.

### Comparison and stop rule

Compare the complete RT retrieval proposer with current MTP, current DFlash2, plain decode,
existing prompt/n-gram lookup where supported, and the same retrieval strategy on CPU/CUDA.
Keep the target artifact, verification semantics, workload and output budget comparable.
Include datastore creation, cold/warm operation, updates, resident memory, misses and capture
overhead. Begin with one chain, avoiding a new target tree-verification requirement.

For equal verification width, a cheaper proposer pays only if its retained acceptance compensates
for any less accurate drafts:

```text
rate_RT = tau_RT / (T_retrieval + T_verify + T_state)
rate_neural = tau_neural / (T_neural_draft + T_verify + T_state)
```

Use measured, mechanism-specific verification/state costs if acceptance or layout changes them.
The index must beat both the neural baseline and a simpler implementation of its own retrieval
algorithm. A fast RT lookup with mostly rejected tokens is a failed proposer experiment.

## Candidate design: RT-assisted neural draft head

```text
Existing MTP/DFlash backbone
            |
Normalized draft hidden states
            |
Approximate RT-assisted vocabulary search
            |
Bounded candidate IDs per position
            |
Score selected rows of the original represented proposal head
            |
MTP/DFlash argmax OR DFlash2 top-16 + existing conditional selector
            |
Existing causal target verification and commit
```

1. Build an immutable search index for the selected proposal head outside the decode loop.
   Start with a JUNO-style subspace/product-quantized inner-product search, not a direct
   projection of every 5120-dimensional row into a single 3D point. Three coordinates alone
   are not a sufficient representation of arbitrary vocabulary rankings.
2. Search for a bounded candidate set per position. Initial experimental budgets can be
   256, 1024, and 4096 unique rows; these are research parameters, not product defaults.
3. Rerank those rows using the original stored representation and the corresponding head
   input. Preserve quantization scales, rotation/Hadamard conventions, token-ID remapping,
   and public-vocabulary bounds. Do not silently use source FP32 weights as the production
   reranker for an artifact that executes quantized weights.
4. Feed the selected IDs and scores to the existing backend-specific proposal logic.
   Define deterministic handling of duplicates, ties, underfilled results, and invalid IDs.
   A failed/underfilled search uses the existing head for that position or block; count and
   time those cases. Ordinary approximate misses are not automatically detectable.
5. Preserve actual proposal probabilities and use the unchanged target correction rule.
   The target output head remains full vocabulary.

The existing indexed-head API supplies a static row map; it does not by itself establish
support for efficient per-position dynamic candidate rows. A native selected-row scoring
operation may be required. That operation is part of the feasibility cost.

For the experimental index, use the selected head's actual represented rows as the source
of search vectors. Any dimension reduction or product quantization is explicitly approximate.
Avoid allocating a permanent full FP32 duplicate. Measure temporary build memory, resident
index memory, scratch, and startup time before choosing a persisted representation.

If the experiment succeeds, converter recipes should produce portable index data tied to the
head's row map and representation. Model owns immutable data and native Parameters; Program
owns scratch, invocation state and CUDA Graphs. Ops own search/scoring contracts, with any
OptiX plumbing beneath those explicit operations. Follow
[Engine architecture](docs/maintainer/engine-architecture.md); do not add a second scheduler,
generic model graph, hidden decode allocations, or runtime weight repacking. Treat a runtime
OptiX acceleration structure as derived execution data; verify its build and memory behavior.

## Correctness requirements

- An approximate proposer may change its guesses and acceptance length. It may not bypass
  the target verifier, commit rejected state, or change public target sampling controls.
- MTP/DFlash currently produce point-mass proposals and use the documented coupled target
  draws. Preserve those semantics when replacing their head search.
- DFlash2 must retain the conditional `q` actually used after candidate selection. For positive
  temperature, use `min(1, p(token)/q(token))` acceptance and the normalized positive residual
  `max(p-q, 0)` on rejection, including zero proposal mass outside the selected support.
  Never report probabilities from the old full candidate set for a different sampled set.
- Greedy proposal misses affect acceptance, not permission to emit an unverified token.
  Do not promise bitwise identity across different verification widths: existing floating-point
  execution boundaries still apply. Distribution correctness and numerical route qualification
  are separate obligations.
- Respect request-local seed/frontier identity, masks, cancellation, tails, compact batches,
  KV/GDN state, and continuation publication as defined in the current DFlash reference.

## Alternatives and constraints

| Approach | Disposition |
|---|---|
| RT vocabulary shortlist plus reranking | Primary hypothesis: static weights permit index reuse and avoided head reads may repay search cost. |
| CUDA-only approximate shortlist | Mandatory competing implementation or credible baseline; distinguishes algorithmic savings from the RT contribution. |
| SpecVocab-style learned predictor | Strong prior art; consider separately if a trained predictor is needed. Training is additional work, not assumed free. |
| Hidden-state continuation retrieval | Secondary research direction only; adds corpus alignment, storage and index lifecycle work. |
| RT acceleration of the 16-way selector | Defer: small, context-dependent work is unlikely to repay an additional ray launch. This remains an inference, not a measurement. |
| RT sparse attention | Outside the first prototype; tracked in the KV/attention research branch below, with a separate numerical contract. |
| Replacement MLPs or candidate-tree traversal | No demonstrated mapping selected. Token trees are not geometric BVHs; these require separate mathematical designs. |

RT cores are not independent free compute. Shader execution, memory bandwidth and cache capacity
are shared with inference. JUNO observed interference under naive co-running. A same-position
head search cannot start before its hidden state exists; subsequent MTP work depends on the
selected token. Do not assume overlap that violates these dependencies.

## Broader research branches and their contracts

### KV selection for attention and data placement

[RetrievalAttention](https://arxiv.org/abs/2409.10516),
[RetroInfer v3](https://arxiv.org/abs/2505.02922v3) and their
[current Microsoft repository](https://github.com/microsoft/RetrievalAttention) supply algorithmic
prior art for attention-aware vector indexes, segmented clustering and coordinated KV placement.
They are not RT implementations. Their CPU attention/offload designs are not an automatic fit
for NInfer's GPU arithmetic ownership. Reuse indexing ideas only through an explicit supported
execution design.

[MagicPIG](https://arxiv.org/abs/2410.16179), with
[code](https://github.com/Infini-AI-Lab/MagicPIG), shows why top-k attention is not universally a
good approximation: aggregate tasks can need contributions from many keys. It motivates sampling
and estimation, not merely increasing search recall. [vAttention v2](https://arxiv.org/abs/2510.05688v2)
offers user-specified probabilistic approximation guarantees. "Verified" here does not mean
bitwise/full-attention equality or exact target token probabilities.

Candidate design: build a reusable index over committed KV blocks during prefill, batch the
current attention queries, refine RT candidates on the GPU, then obtain pages through existing
residency and transfer mechanisms. Include appended keys, growing contexts, eviction, forks and
restores in the index lifecycle. Immutable committed segments plus a small directly scanned
recent tail are a hypothesis for limiting rebuild cost, not an implemented layout.

First evaluate this on the drafter's attention, where approximation can change proposals while
the full target still verifies. The short local DFlash2 window may leave little work to save;
keep the unmodified drafter as the baseline. Sparse target attention is a separate numerical
product change. Verifying a proposal with an approximated target does not restore the original
target distribution. Measure representative per-layer/head errors and downstream failures,
index build/update latency, total memory and end-to-end inference before considering adoption.

### Exact target-head search

A conservative candidate superset could support exact greedy selection, but ANN recall does not
certify that no excluded row wins. One real-arithmetic research bound for a cluster with center
`c` and radius `rho >= max_v ||w_v-c||` is `dot(c,h) + rho*||h||`. A cluster can be excluded
only if a safely evaluated upper bound is below the current winning score. This follows from
Cauchy-Schwarz; it is our mathematical starting point, not a demonstrated RT implementation.
Custom per-query high-dimensional bounds do not automatically map to hardware BVH traversal.
Quantized row decoding, floating-point rounding and stable tie rules require conservative error
allowances before such a certificate could preserve the actual execution contract.

Even certified greedy top-1 or top-k does not solve arbitrary positive-temperature sampling:
softmax normalization, penalties, masks and filters can depend on excluded tokens. Preserve the
full target path until a separately proven exact sampling method covers those controls.

### Load/store, media and other accelerators

For page/region queries, require exact metadata results, including boundary and large-integer
offset behavior. Mapping 64-bit offsets into floating-point coordinates without a conservative
encoding can lose matches. With only a few requests, a CPU interval lookup may finish before a
GPU launch; no cache-index bottleneck is presently established.

Video motion estimation is a separate hardware hypothesis. NVIDIA exposes
[NVENC motion-estimation-only mode](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-video-encoder-api-prog-guide/index.html#motion-estimation-only-mode)
and an [AppEncME sample](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/read-me/index.html).
One could encode contextual signatures into image blocks and turn match offsets into draft
continuations. The open questions are similarity preservation, spatial search coverage, supported
formats and complete packing/submission/synchronization cost. This does not directly perform
neural prefill or DFlash arithmetic, and no direct speculative-decoding demonstration was found
in the bounded search. It belongs to a distinct experiment, not the RT prototype's scope.

The main uncertainty is latency at small query counts. DFlash2 with seven draft tokens supplies
seven head queries for one request; MTP often supplies one at a time. Evaluate concurrency one
first, then two and eight. Paper-scale batched QPS does not resolve this question.

Use NVIDIA's current [OptiX documentation](https://raytracing-docs.nvidia.com/optix9/guide/optix_guide.250130.LTR.pdf)
and [acceleration-structure API](https://raytracing-docs.nvidia.com/optix9/api/group__optix__host__api__acceleration__structures.html)
when implementing. Confirm the then-current stable SDK, driver/toolchain compatibility,
redistribution terms, stream dependencies and graph capture/replay support for the actual calls.
Do not assume OptiX support from ordinary CUDA Graph support or install an SDK for this document.

## Bounded experiment and decision criteria

No experiment has run for this proposal. Complete the implementation batch before its checks;
do not turn these phases into per-edit test runs or an open-ended benchmark campaign.

### 1. Establish the opportunity

After the Windows baseline is ready and the GPU is available, collect one representative
baseline covering prose, code/editing and multilingual output, at short and long supported
contexts. Record artifact path/representation, GPU, driver/toolchain, backend, draft count,
sampling, concurrency, head choice and workload. Use existing profiling where sufficient.
Separate backbone, head, selector, verification, synchronization and complete-round time.

Use the configured optimized head as the primary baseline. Record full-head behavior as a
secondary comparison, including its different memory cost. Qualify DFlash2 and MTP separately.

For total round cost `C`, accepted published outputs `tau`, affected head fraction `f`,
head acceleration `s` including all search/rerank costs, and retained acceptance ratio `r`:

```text
output rate = tau / C
estimated relative rate = r / (1 - f + f/s)
```

This estimate assumes other costs stay constant and is only a screening model. For example,
`f=0.20`, `s=2`, and `r=0.90` gives no net speedup. Even eliminating the head cannot save
more than its share of round time at unchanged acceptance. Stop if that upper bound cannot
justify the implementation and memory costs.

### 2. Implement and compare candidate mechanisms

Use identical captured draft hidden states and represented head rows to compare the existing
head, CUDA shortlist plus reranking, and RT shortlist plus reranking. Include index lookup,
query projection, candidate deduplication, row reads, top-k merge, launches and synchronization
in timing. Record startup/build time and steady-state latency separately.

Use independently decoded quantized weights and a naive FP32/FP64 dot-product oracle for the
selected-row operation, with the representation's defined casts and numerical criteria.
Search is intentionally approximate: report top-1/top-16 recovery, missing target tokens,
score margins, result underfill and workload-specific failures rather than demanding fictitious
exactness. Also compare against the actual current head route, since arithmetic ordering may
move near ties. Approximate search recall alone cannot demonstrate output throughput.

Do not integrate a candidate that fails to reduce affected-path latency at useful acceptance
quality. Distinguish a CUDA algorithm win from an additional hardware-specific RT win.

### 3. Validate the complete proposer

For a viable candidate, run a single planned integration qualification batch over the baseline
workloads. Measure accepted tokens per round, full round time, output tokens/s, request latency,
index/scratch/total VRAM, and startup time. Cover greedy and positive-temperature DFlash2;
check actual-q correction against a small independent probability oracle and state commit
behavior at rejection and tail boundaries. Test deterministic IDs/remapping and resource lifetime.

Use interleaved comparisons and enough repeats to distinguish the candidate from observed
variation. Report each workload, spread, worst slowdown, underfill/fallback frequency, and
out-of-vocabulary/acceptance effects. Keep context, sampling, draft count, concurrency and
artifact fixed for the primary comparison. If a different draft count is subsequently chosen,
report that as a separate system configuration comparison.

Adopt only if end-to-end gains exceed measured variation for the intended workload mix,
correctness holds, VRAM fits without silently reducing context/concurrency, and startup and
maintenance costs are acceptable. Any workload-specific win must be labeled as such, with
regressions visible. Do not change acceptance criteria after seeing results to manufacture a win.
If CUDA wins without RT, keep that result and reject the RT addition. If neither wins, close
the experiment and remove the temporary implementation.

## Completion and next action

This proposal is complete when the sources, hypothesis, integration boundary and decision
criteria are reviewable. Implementation, SDK installation and performance qualification remain
future work. The first implementation decision depends on the measured 4090 head fraction;
no expected percentage gain is asserted here.

Next research questions, in order:

1. Does a bounded, norm-preserving MIPS shortlist recover useful draft candidates on the actual
   represented 27B head? Compare low-dimensional screening, subspace/PQ and a CUDA alternative
   before choosing RT geometry. No public ready-made implementation reviewed here answers this.
2. Which work can use built-in triangle intersection, and which needs custom programmable
   intersection/refinement? Count the latter as CUDA work and measure primitive/launch overhead.
3. Can a persistent index and bounded device-only candidate buffers integrate with the existing
   CUDA Graph and Program lifecycle using the selected current SDK? Establish support rather
   than deriving it from a research application's asynchronous launch method.
4. Where does long-context time actually go in this hybrid model? Establish full-attention versus
   GDN/FFN cost and KV traffic before escalating the broader attention branch.

Update this same document when evidence answers those questions. A rejected mapping, slowdown,
quality loss or memory cost remains part of the decision record; do not replace it with a more
favorable baseline or omit it from the result. This document is not a background monitor.

When the experiment concludes, place durable supported behavior in the existing DFlash/model
references and measured results in the existing performance authority. Remove this temporary
root proposal when completed or abandoned rather than maintaining parallel architecture docs.
