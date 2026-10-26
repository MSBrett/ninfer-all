# Quality trades

The precision options of `ninfer`, `ninfer-serve` and `ninfer-perplexity`: what each stores or
computes differently, what it costs in quality and what it buys. The switches below are off by
default except the integer prefill routes, which `--no-prefill-a8` turns off. Dense Qwen3.8-27B
artifacts can also permit floating FP8 activation compute for their FP8/NVFP4 weights; `--no-fp8-compute` caps those
Uses at A16 independently of integer activation controls and KV storage. Unless a row says otherwise the
measurements are Qwen3.8-27B (`groupwise-int`) on one RTX 3090, perplexity from
`ninfer-perplexity --quick` on `ninfer-ppl-1m-v1` (context 4,096, stride 2,048).

| option | what changes | quality | gain |
|---|---|---|---|
| `--embedding-q4` | token embedding stored as Q4 G64 at load | 4.346413 → 4.343738 (−0.062%, noise) | −644 MiB of weights, about 24K tokens of `rk8v4` context |
| `--embedding-q6` | token embedding stored as Q6 G64 | Qwen3.6-35B-A3B: +0.035% | −341 MiB (27B), −143 MiB (35B-A3B) |
| `--lm-head-q6` | output head stored as Q6 G64 | +0.012% | −341 MiB, about 13K tokens |
| `--lm-head-q4` | output head stored as Q4 G64 | **+0.693%**, greedy text parts within ~50 tokens | −644 MiB; +3% aggregate decode at eight lanes |
| `--mtp-experts-q4` | Qwen3.6-35B-A3B: the MTP layer's routed experts in the text layers' formats | none scored (drafts are verified); acceptance 61.2% → 60.6% | about −350 MiB, two rungs of context |
| `--gdn-state-fp16` | GDN recurrent state stored as FP16 between steps, computed in FP32 | no perplexity change; per-step state drift at most 0.34% | halves every StateImage (27B: 147 → 74.5 MiB); +2% aggregate decode at eight lanes |
| `--prefill-cublas` | wide prefill GEMMs as int8 weights with one scale per row through cuBLAS | +0.156% (+0.072% with `--no-prefill-cublas-projections`) | prefill 43-83% faster; workspace 155.6 → 661.2 MiB at chunk 4096 |
| `--mlp-a8-decode` | MLP gate/up at 16-32 verify columns with int8 activations | oracle bound 0.008-0.037 relative L2; perplexity cannot see it | +1.3% aggregate decode at eight lanes |
| `--no-prefill-a8` | returns full prefill tiles to their A16 routes | the integer routes cost +0.004% | the integer routes prefill 49-67% faster |

`--embedding-q4` and `--embedding-q6`, and `--lm-head-q4` and `--lm-head-q6`, are mutually
exclusive. A Q6 flag is a no-op where the artifact already stores Q6 (the Qwen3.6-27B embedding and
head, the Qwen3.6-35B-A3B head; the head flags are rejected on that model).

## Vocabulary transcoding

`src/artifact/transcode.{h,cpp}` requantizes a row-split `q8_g32_fp16` payload into a narrower
group-64 row-split encoding on the host while the weights load. The binder plans the target
encoding's size, so the arena shrinks, and the materializer writes the target bytes before anything
else (bindings, the overlay eviction mirror) observes the tensor. Each group's FP16 scale is the
best of 25 clipping ratios of `absmax / qmax` in [0.70, 1.18], so a group is never worse than
round-to-nearest. It is multi-threaded and adds a few seconds to load.

- The embedding of the 27B quantizes for free; the 2048-wide embedding of the 35B-A3B does not:
  `--embedding-q4` costs it +0.36%, about what NVFP4 KV costs, so `--embedding-q6` is the option
  there.
- `--lm-head-q4` and `--lm-head-q6` are rejected with DFlash and DFlash2, whose candidate top-k reads
  the stored head directly.
- Measured before the Q6 vocabulary-head small-T kernels existed, `--lm-head-q6` cost 2-5% of
  one-lane decode; the device profiles now route the Q6 head, and that cost has not been measured
  again.
- Overlay Vision borrows its exclusive encode window from the evict-ranked weights (head,
  embedding, draft head, MTP), so they must cover one window. On the 35B-A3B a smaller embedding
  can fall below the 960 MiB window of `--vision-max-merged 16384`; startup then refuses and names
  the ways out (a lower `--vision-max-merged`, `--vision-residency resident`, or dropping a flag).
  The eviction mirror is captured after transcoding, so overlay windows restore transcoded bytes.

Qualification: `ninfer_vocabulary_transcode_test` checks the codec against an independent decoder
and routes transcoded 248320x5120 heads through `ops::linear` at T = 1..32; `ninfer_embedding_test`
qualifies the Q4 gather at [248320,5120] and [248320,2048] against an FP64 oracle, CUDA Graph
replay included.

## FP16 GDN state

The 27B's GDN recurrent state is 48 layers x 48 heads x 128 x 128 FP32, 147 MiB per sequence slot,
read and written every round. With `--gdn-state-fp16` the recurrence still computes in FP32 and only
the stored state between steps is rounded: the four recurrent kernels (direct, batch update,
record, replay fold) are templated on the storage type, and the chunked prefill kernels keep their
FP32 interface with the wrapper staging FP16 through an FP32 workspace.

Perplexity scores through prefill, which rounds the state only at chunk boundaries, so it cannot
clear this option alone. `ninfer_gdn_state_fp16_test` runs 4,096 synthetic decode steps through the
real Op with FP16 against FP32 state: the worst relative error is 0.339% at width 1 and 0.219% at
width 4, flat over the run rather than growing. Greedy output on a real prompt was byte-identical.

## Integer activations

The integer prefill routes (`src/ops/common/rowsplit_a8_mma.cuh`) quantize activations per token
and 64-column group and run `mma.m16n8k32.s8`; they are on by default because their cost is inside
run-to-run noise (+0.004%) and their FP64 oracle bound is 0.010-0.020 relative L2 against the
0.04 allowance A8 activation compute is held to. A per-token scale, as Marlin-class kernels use,
would be about 20% faster but reaches 0.129 relative L2 on outlier-heavy inputs.

`--prefill-cublas` goes further for wide prompts: the weights are materialized as int8 with one
scale per row and the GEMM runs in cuBLAS. cuBLAS reduces over the whole of K, so the per-group
scales are lost; equalizing the activation channels (a per-channel scale folded exactly into the
weights) is what keeps the cost at +0.072% for the MLP and output projections instead of +0.474%.
It wants a large `--prefill-chunk`, since its dequantization pass is weight-sized and its GEMM
token-sized; the trade-off per chunk is in `src/ops/linear_swiglu/q4cublas/w4_cublas_prefill.h`.

A rejected Q4 gate/up `[34816,5120]` experiment combined offset RMSNorm with collection of
the normalized activation's channel maxima, retaining BF16 hidden storage, rounding and the
existing full-chunk INT8 equalization, quantization, cuBLAS GEMM and SwiGLU. On RTX 4090 with
CUDA 13.4.3, all six complete-stage comparisons at T=512/1024/2048 with warm and cold cache
were slower: mean overheads ranged from 1.62% to 4.30%, with a worst replicate of 4.49%.
At the saved T=1024 chunk size, warm time was 1145.295 to 1189.760 microseconds and cold
time was 1132.030 to 1177.090 microseconds. These are stage measurements, not whole-inference
results; see the [timing evidence](../../profiles/bench/rmsnorm-int8/bench/initial/summary.json).

Exact preservation of intermediates and output passed, but the unchanged independent FP64
criterion failed on forced T=257 replay 1, output index 87: 260 versus 270.857. The staged
baseline produced the same failing output; its existing cuBLAS regression program passed.
Full numerical qualification therefore did not pass; see the
[qualification evidence](../../profiles/bench/rmsnorm-int8/qualification/initial/preservation-findings.json)
and [program results](../../profiles/bench/rmsnorm-int8/qualification/initial/summary.json).
The built-in RTX 4090 profile has no cuBLAS gate override, so its default T>=512 gate does not
select cuBLAS at T=257; that diagnostic explicitly forced the route. The automatic model
integration was removed. There is no shipped RMSNorm/INT8 fusion, speedup or VRAM saving
from this experiment.

`--mlp-a8-decode` extends the integer gate/up route to the 16-32 columns a batched verification
round reaches (eight lanes of four MTP columns is 32); narrower widths stay on the BF16 small-T
kernel, which is faster there. Perplexity scores the prefill phase and never reaches this
verify-phase route, so its evidence is the oracle bound and output comparison at eight lanes. The
kernel's measurements and rejected variants are in the header of
`src/ops/linear/q4/q4_small_t_mma_i8.cuh`.

## Dense Qwen3.8-27B on Ada

An A8 permission allows route selection; it does not require FP8 compute for every projection.
Ada's row-scaled FP8 Linear `[5120,6144]` uses A16 through T=32 and A8 from T=33 by default.
Packed NVFP4 currently defaults to A16 for Linear, LinearAdd and LinearSwiGLU; its A8 candidate
remains available through the finite [device-profile keys](../device-profiles.md#dense-qwen38-27b-activation-routes-on-ada).
Both compute paths retain the compressed artifact weights.

The dense Qwen3.8-27B SM89 NVFP4 A8 projection route keeps the artifact's packed E2M1 codes,
E4M3 group scales and original weight divisors. Public BF16 activations widen exactly to FP32.
A token uses a midpoint center
only when all its values are strictly positive (`min > 0`) or strictly negative (`max < 0`):
`c = RN32(RN32(min/2) + RN32(max/2))`. Otherwise `c = 0`, retaining the original rowwise
max-abs FP8 quantization, including rows that touch zero. FP32 round-to-nearest-even computes
the scale as `s = RN32(max(abs(RN32(min-c)), abs(RN32(max-c))) / 448)`.
Residuals use FP32 subtraction and multiplication by the rounded reciprocal of `s`;
a nonfinite reciprocal takes FP32 division
instead. E4M3 conversion rounds to nearest-even with finite saturation. Constant rows keep their
center with zero scale and residual codes.

Unpacked E2M1 weight codes are exactly representable in E4M3. The product restores the removed
center with the weighted E2M1 code sum: each G16 code sum is exact in integer half-units before
its FP32 scale-weighted reduction. With the weight row's own global divisor `d_w`, the normal
epilogue forms `ratio = RN32(s/d_w)` and `center_ratio = RN32(c/d_w)`, then
`FMA32(sumQ, ratio, MUL32(sumW, center_ratio))`. Zero-centered tokens skip the center correction.
Blocks whose live centers are all zero also omit code-sum reductions and weighted accumulation.
LinearAdd first fuses a nonzero center correction with its BF16 residual, then adds the scaled
dot product with one FMA. A nonfinite or subnormal ratio
for a nonzero coefficient widens only the scalar affine rescale to FP64; MMA and stored operands
remain unchanged. There is no expanded persistent weight copy. SwiGLU receives the corrected
gate/up values before its nonlinearity. This centering belongs to NVFP4 A8; the A16 arithmetic
and the row-scaled FP8 activation quantizer are separate paths.

The center buffer adds one FP32 word per token (`4*T` bytes before 256-byte workspace alignment)
to the prior row-scale A8 workspace. This is a private compute representation; the numerical oracle
still starts from the original BF16 activations and independently decoded stored weights, with
the existing FP64 Op formula and error criteria. The oracle does not reproduce the activation
quantizer or center correction.

The earlier one-sided NVFP4 A8 arithmetic passed four affected GPU qualification programs on
the RTX 4090 with no skips, against the original BF16-input FP64 oracles and unchanged criteria.
The subsequent combined batch has not passed: new FP8 biased-input oracle and NVFP4 fused
validation failures require repair and requalification. The existing
[CPU representation diagnostic](../../profiles/bench/ada-fp8-nvfp4-cuda134/cpu-one-sided-midpoint-design.json)
remains a diagnostic over its recorded samples, not a replacement for GPU qualification or a
measurement of model quality or performance. Earlier centering regressions remain preserved in
the experiment reports.

The [initial cold-cache comparison](../../profiles/bench/ada-fp8-nvfp4-cuda134/micro/initial/)
found NVFP4 A8 slower in 19 of 20 Linear cases and at small fused widths. Its wide fused gains
were relative to a slow A16 fallback, not an optimized A16 baseline. The replacement A16 SwiGLU
reuses the fast Linear mainloop with an FP32 collective epilogue; A16 LinearAdd above T=32
reuses the existing optimized A16 launcher. Their combined numerical and performance
qualification has not completed, and final timing has not run. These Op measurements establish neither final model throughput nor
a KV-capacity gain.

### Attention normalization fusion

The first dense-model integration of fused RMSNorm and FP8 activation quantization is the
primal, single-parent attention projection `[14336,5120]` on a native RTX 4090/SM89 image,
when its selected table and Use policy admit A8. The producer retains the ordinary offset
RMSNorm's BF16 rounding before calculating the full-token FP8 scale and affine center. The
existing A8 projection consumes those prepared operands and writes the same Q/gate/K/V
buffers, without materializing the normalized BF16 hidden tensor. Other routes retain the
staged normalization and projection.

The Op supports an optional BF16-rounded residual sum, but this model call already receives
the preceding residual update in `x`; it does not split an existing LinearAdd to repeat that
addition. GDN keeps its existing fused norm/control path. The attention call uses the ordinary
A8 scratch contract. Startup conservatively retains the normalized-hidden reservation for
unfused widths and device/profile choices, so omitting that allocation during fused execution
does not establish a total VRAM saving.

On 2026-10-25, the SM89/CUDA 13.4.3 build and the dedicated fusion test passed on the RTX 4090
(1.345 seconds, zero skips). The test covers T=5/17/33/65/1024, optional residual addition,
changed-input CUDA Graph replay, full-row equality of FP8 codes/scales/centers, exact staged
projection output equality, and an independent FP64 composition oracle. The first build needed
an explicit CUDA math-constants include; that integration error was corrected before execution.
Evidence is in `profiles/bench/ada-fp8-nvfp4-cuda134/qualification/fp8_rmsnorm_essential/`.
Broader FP8 regression runs and the prepared fused-versus-staged benchmark are deferred while
the GPU serves the user's normal model. No inference-speed or KV-capacity gain is claimed.
