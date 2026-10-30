# Bonsai 2 27B (ternary PQ2_0) on ggml-openvino with OpenVINO PR #38460

Running `prism-ml/Ternary-Bonsai-2-27B-PQ2_0.gguf` through the llama.cpp OpenVINO backend on
top of the TernOCL int2 GPU kernels from openvinotoolkit/openvino#38460, and comparing against
ggml-vulkan on the same GPU.

## 1. Setup

| | |
|---|---|
| llama.cpp | `PrismML-Eng/llama.cpp` branch `prism` (carries the PQ2_0 type and the Hadamard runtime; stock llama.cpp cannot read these files), working branch `prism_ov_ternary` |
| ggml-openvino | replaced wholesale with `ravi9/llama.cpp` `dev_backend_openvino` @ `a4493f9c` |
| OpenVINO | `openvinotoolkit/openvino` PR #38460 @ `5c6d4692ee`, built from source + TernOCL submodule |
| Model | 6.70 GiB, 26.90 B params, 2.13 bpw, arch `qwen35` (dense, hybrid GDN + full attention, 64 layers) |
| GPU | Intel Arc B390 iGPU (Xe3), 58 GB shared |

The graft in step 2 was clean. The `prism` base is on `GGML_BACKEND_API_VERSION 2` and the ravi9
branch targets 3; the only adaptation needed was dropping the two API-3-only
`ggml_backend_buffer_type_i` slots (`alloc_buffer_n`, `get_alloc_size_n`) from the positional
initializers, which the backend left `NULL` anyway. Nothing else conflicted.

Caveat on the hardware: PR #38460 tunes its tile tables for Xe2 (discrete and integrated). This
GPU is Xe3, so any shape outside those tables takes an untuned default tile. The kernels still run
correctly; the absolute numbers below are not what a tuned Xe2 part would give.

## 2. What had to change in ggml-openvino

Four changes, in `ggml/src/ggml-openvino` (commits `50c37b9e`, `b1d205be`).

**1. PQ2_0 -> OpenVINO `u2`.** The ggml block is an f16 scale plus 32 bytes holding 128 ternary
codes, code `j` at bits `[2*(j%4), +1]` of byte `j/4`, dequantized as `(code - 1) * scale`. That is
bit-identical to how OpenVINO packs `u2` (everything except `u1` is LSB-packed), so extraction is a
verbatim `memcpy` of `qs` plus gathering the scales - no bit shuffling. The emitted chain is
`Constant(u2) -> Convert(f16) -> Subtract(1) -> Multiply(scales)`, which the GPU plugin folds into a
compressed `FullyConnected`. The zero point is one scalar for the whole tensor and is emitted as an
f16 `Constant`, not a `Convert` of an integer one, because the kernel reads it directly.

**2. Joint q/k L2 norm in dense `qwen35`.** `src/models/qwen35.cpp` normalizes q and k as a single
`[2 * n_k_heads]` tensor and hands `GATED_DELTA_NET` views into the result. A `VIEW` is a
pass-through on this path (the contract is that consumers re-slice), and the GDN translator only
re-sliced its `v` input, so q and k both arrived at the joint 32-head width and the GQA tiling then
inflated them to 96 against 48 value heads. Fixed by re-slicing q/k on the head axis. `qwen35moe`
(Qwen3.5 / 3.6 35B-A3B) normalizes q and k separately, which is why this had never been hit - it is
an unimplemented case, not a regression.

**3. RESHAPE dynamic-dim propagation across a scaled axis.** The propagation only recognised an
output axis that *keeps* the extent of the source dynamic axis. The blockwise Hadamard rotation
reshapes `[T, K] <-> [T * K / 1024, 1024]`, which scales it, so the axis read as static and the
captured token count was baked into the reshape target: `Requested output shape [1,1,10,1024] is
incompatible with input shape [1,1,42,5120]`. Added a span-based fallback, and made reshape
`op_case 0` honour `get_op_dynamic_dim()`.

**4. Ternary weights must skip requantization.** `ggml_openvino_get_requant_type` sends
`token_embd.weight` and `output.weight` to `Q8_0_C` unconditionally. Every requant target is 4-bit
or wider, so on a ternary model this inflates 2.13 bpw weights to 8 and drops `output.weight` off
the ternary path.

### The one non-obvious interaction

These are not independent. Bug 3 was the root of a failure that *looked* like a kernel-acceptance
problem. The baked reshape made the FC input shape static, which failed
`FullyConnectedHorizontalFusion::is_target_pattern`'s `is_dynamic()` requirement, so gate/up never
merged, so `ffn_gate` kept a three-op fused chain that matches none of TernOCL's five epilogues.
Because that FC also carried a fused Hadamard transform, the reject became a hard build failure
(by design - no other implementation applies the rotation). Fixing the reshape made horizontal
fusion fire (`ffn_gate_fused_2FCs`, `z_fused_2FCs`, `Qcur_full_fused_3FCs`) and the epilogue match.

## 3. Correctness

The only remaining TernOCL rejects are the `ssm_alpha` / `ssm_beta` projections, whose weights are
BF16 rather than ternary; they correctly fall through to the stock path. Every u2 FC is accepted.

- OV GPU output is **identical to the ggml-CPU ground truth** on a short greedy prompt
  (`"The capital of France is"` -> same reasoning trace, `Paris`).
- OV GPU default and int8 prefill (`OV_GPU_TERNOCL_INT2_INT8_PREFILL=1`; `OV_TERNOCL_INT2_INT8_PREFILL=1` on builds up to `5c6d4692ee`) are **byte-identical** over 90 greedy tokens.
- Vulkan agrees semantically but diverges in wording after ~35 greedy tokens. This is ordinary
  near-tie drift between different kernels, not a correctness failure.
- `test-backend-ops -b OPENVINO0`: **8079/8084**. All 5 failures are `GATED_DELTA_NET`
  (`raw_gates=1` / `rows_mode=1`) and reproduce identically on the pristine pre-change graft build,
  so they are pre-existing on `dev_backend_openvino`.
- llama-3.2-1B and gemma-4-E2B still produce correct output at unchanged speed on OV GPU, so the
  global reshape change did not regress other models.

## 4. Performance

Same GPU, same model file, t/s. Every row is on **OpenVINO master** (`5d07300df3`, 2026-10-30),
which now contains PR #38460 (merged as `f4e23bfdf1`), with OpenVINO GenAI master (`49237dab`)
built in. ggml-openvino is this branch **after syncing with `ravi9/dev_backend_openvino` @
`02d71ff9e`**. Decode is measured with `-n <N> -r 5` **on its own** and prefill with
`-p 128,512 -r 3`; the measurement note at the end of this section explains why.

| configuration | pp128 | pp512 | tg32 | tg128 |
|---|---:|---:|---:|---:|
| **OpenVINO GenAI** (`VLMPipeline`, `llm_bench`) | 258.1 | **680.0** | **14.64** | **14.57** |
| **OpenVINO GenAI**, int8 prefill | **668.6** | **740.4** | - | - |
| **ggml-openvino**, `ENABLE_KV_4D=1 KV_SLICE=0` | 236.9 | 536.1 | 13.27 | 13.28 |
| **ggml-openvino**, default settings | 236.2 | 541.4 | 13.14 | 13.16 |
| **ggml-openvino**, int8 prefill | 546.8 | 570.9 | - | - |
| ggml-openvino before the sync (same OpenVINO) | 236.3 | 544.7 | 13.19 | 13.23 |
| **ggml-vulkan** | 171.2 | 169.5 | 12.70 | 12.70 |
| ggml-openvino, TernOCL disabled | 1.4 | - | 1.24 | - |

"int8 prefill" means **`OV_GPU_TERNOCL_INT2_INT8_PREFILL=1`**. The old
`OV_TERNOCL_INT2_INT8_PREFILL=1` is **silently ignored** by current builds (verified: pp128 stays at
the default). Builds up to the PR revision `5c6d4692ee` read only the old name.

GenAI / ggml-openvino: pp128 **1.09x**, pp512 **1.27x**, decode **1.10x** (int8 prefill on both:
1.22x / 1.30x). ggml-openvino / Vulkan: pp512 **3.16x** (3.37x with int8 prefill), decode **1.05x**.

**The sync is neutral on this model.** Against the pre-sync branch on the same OpenVINO, decode
moves 13.23 -> 13.28 (+0.4%) and prefill stays within noise. The 69 synced backend commits target
other models, NPU and `-fa off`; none touches the GDN/ternary path's hot ops. Two side effects:
the defaults are now almost as fast as the tuned settings (13.16 vs 13.28), and
`GGML_OPENVINO_DISABLE_KV_SLICE` is gone. Its replacement, `GGML_OPENVINO_KV_SLICE` (default 1),
no longer affects Bonsai decode (13.28 at both 0 and 1).

**OpenVINO master vs the PR head:** ggml-openvino is unchanged (13.28 vs 13.21 t/s tg128, separate
sessions), while GenAI prefill improved (pp512 658 -> 680, int8 715 -> 740).

The GenAI rows are real `openvino_genai.VLMPipeline`, run with the PR doc's `llm_bench` recipe on
the u2 IR plus the Qwen3.5 tokenizer, config and vision encoders from the template export (Bonsai 2
is a Qwen3.5 VLM to GenAI; text prompts never run the vision encoders). Prefill t/s is prompt
tokens / first-token latency at an exact 128- or 512-token prompt (`genai_prompts/p{128,512}.jsonl`,
verified by llm_bench's reported input size); decode is llm_bench's 2nd-token throughput. Output
text was checked coherent on every build. The whole table is reproduced by `bench_master.sh`.

#### The same comparison on the earlier PR build (`5c6d4692ee`)

That revision shipped no GenAI path. Its native reference was `paged_bench_llm_27b`, the PR's own
PagedAttention tool, which has since been removed from the PR.

| configuration | pp128 | pp512 | tg32 | tg128 |
|---|---:|---:|---:|---:|
| native OpenVINO (`paged_bench_llm_27b`) | 241.2 | 638.5 | 14.90 | 15.03 |
| native OpenVINO, int8 prefill (`OV_TERNOCL_INT2_INT8_PREFILL=1`) | 616.3 | 693.4 | - | - |
| ggml-openvino, final | 236.1 | 535.1 | 13.45 | 13.46 |
| ggml-openvino, int8 prefill (`OV_TERNOCL_INT2_INT8_PREFILL=1`) | 549.0 | 579.7 | - | - |
| ggml-openvino before this work | 232.2 | 527.3 | 12.27 | 12.24 |

Its 21-token reference run reproduces the doc's published `generated_ids` exactly, which is the
end-to-end check that the GGUF -> IR weight mapping is correct.

### Prefill

ggml-openvino is **3.16x ggml-vulkan at pp512** (536 vs 169), and **3.37x** with int8 prefill
(571). Vulkan is flat from pp128 to pp512 (171 -> 169) while OV scales (237 -> 536), which is what a
tiled DPAS GEMM should do against a per-token kernel.

Int8 prefill is off by default in the PR but is a large win at short prompts: **2.3x at pp128**
(237 -> 547), 1.06x at pp512. Greedy output was byte-identical to the default over 90 tokens.

Against GenAI, ggml-openvino prefill is close at short prompts (1.09x at pp128) and further behind
at pp512 (1.27x; 1.22x / 1.30x with int8 prefill on both). The pp512 gap widened on OpenVINO master
because GenAI's prefill improved and ggml-openvino's did not. Even so, the larger remaining
integration gap per token is decode.

### Decode

ggml-openvino and Vulkan tie at ~12 t/s. **Native OpenVINO does not** - it sustains 15.0 t/s on
the same iGPU and the same weights, **1.22x** ggml-openvino's clean 12.25.

This **corrects the earlier conclusion** that ~80 GB/s was "roughly all this iGPU can sustain" and
that there was "no headroom to win here". The measured memory system is 8 x 16-bit LPDDR5X-9600 =
**153.6 GB/s theoretical peak**, so the numbers are:

| | t/s | implied traffic | % of 153.6 GB/s peak |
|---|---:|---:|---:|
| native OpenVINO | 15.03 | ~103 GB/s | 67% |
| ggml-openvino | 12.25 | ~88 GB/s | 57% |

~88 GB/s was an *integration* ceiling, not a hardware one. **See §4.1 for the root cause** - the
whole-token averages above are misleading: the ternary GEMV kernels themselves reach 106 GB/s in
*both* backends, and the apparent deficit is entirely time spent on non-GEMV ops that move almost
no data.

A discrete part is still needed to characterise the kernels themselves, but it is no longer true
that decode cannot distinguish the backends on this hardware - it can, and does.

> **Retracted:** an earlier revision of this section claimed ggml-openvino decode *degrades with
> depth* (12.25 at tg32 -> 11.28 at tg128) while native OV stays flat, and that the gap therefore
> widens with context. That is **wrong** - the 11.28 came from the `pp`-before-`tg` ordering
> artifact described in the measurement note below. Measured on its own, ggml-openvino decode is
> flat (12.27 / 12.24), exactly like native OV. There is no depth-degradation effect here.

### 4.1 Root cause of the decode gap: non-GEMV graph overhead, not bandwidth or dispatch

Measured, not inferred. Two independent instruments:

**GPU occupancy (xe PMU, `gpu_busy.py`** - `perf` is not built for this kernel, so the counters are
read through `perf_event_open`; validated at 0% idle / 96.4% during prefill):

| | compute/CCS busy | copy/BCS busy | t/s |
|---|---:|---:|---:|
| ggml-openvino decode | **99.99%** | 14.1% | 12.8 |
| native OpenVINO decode | **99.95%** | 12.5% | 14.9 |

Both saturate the GPU. So the gap is **not** host dispatch, not kernel-launch gaps, and not GPU
idle - which **rules out** the `update_implementation`-style dispatch bottleneck found on gemma-4
MoE. ggml-openvino simply asks the GPU to do more work per token.

**Per-op breakdown** (`GGML_OPENVINO_PROFILING=2`, steady-state decode, 2718 executed ops):

| op | ms/token | share | count |
|---|---:|---:|---:|
| `FullyConnectedCompressed` (TernOCL ternary GEMV) | 64.72 | 84.4% | 257 |
| `Transpose` | 2.80 | 3.7% | 273 |
| `Convert` (mostly `reorder_data_fast_b1__f32`) | 2.52 | 3.3% | 275 |
| `GatedDeltaNet` | 1.87 | 2.4% | 48 |
| `Result` (`cache_s_l*` state writeback) | 1.06 | 1.4% | 96 |
| `FullyConnected` (the BF16 `ssm_alpha`/`ssm_beta`) | 0.90 | 1.2% | 97 |
| `RMS` | 0.70 | 0.9% | 209 |
| `StridedSlice` | 0.39 | 0.5% | 336 |
| rest (`Concat`, `SDPA`, `Add`, `Reorder`, ...) | ~1.0 | 1.3% | - |
| **total** | **76.70** | | |

The decomposition closes exactly:

```
ggml-openvino   64.72 ms GEMV + 11.98 ms other = 76.70 ms  (13.0 t/s)
native OV                                        67.19 ms  (14.9 t/s, measured)
gap 9.51 ms   <->   non-GEMV overhead 11.98 vs an implied ~2.47 ms for native
```

**The ternary GEMVs are not the problem.** 6.86 GB of weights in 64.72 ms is **106 GB/s**, 69% of
the 153.6 GB/s peak, and both backends run the *same* TernOCL kernels over the *same* u2 weights,
so that term is identical on both sides. The entire 1.17x is the ~9.5 ms/token of graph scaffolding
that the hand-built native graph does not have.

#### The single biggest item: the GDN recurrent-state round-trip

Breaking the overhead down by *cause* rather than by op type makes one mechanism dominant. Three
different op types all appear exactly **96 = 48 GDN layers x 2 (state in + state out)** times:

| per-token cost | op | n | ms |
|---|---|---:|---:|
| state layout swap `{0,1,3,2}` | `Transpose` / `permute_xy_swap__f16` | 96 | 1.73 |
| state dtype conversion to f32 | `Convert` / `reorder_data_fast_b1__f32` | 96 | 2.43 |
| state writeback | `Result` (`cache_s_l*`) | 96 | 1.06 |
| | | | **5.22** |

That is **44% of all non-GEMV overhead and 6.8% of decode**, spent entirely on shipping the GDN
recurrent state across the graph boundary and back every single token. The state is
`[1, 48, 128, 128]` f32 = 3.1 MB per layer, ~151 MB per token in total.

Why it happens: `translate_gated_delta_net` receives the state in ggml's layout
`[B, H_v, value_dim, key_dim]`, but the fused `GatedDeltaNet` op wants
`[B, H_v, key_dim, value_dim]`, so it transposes on the way in (line ~181) and transposes straight
back on the way out (line ~220). The dtype conversions are the same boundary: ggml keeps recurrent
state in f32 while the model body runs f16.

**The layout half is now fixed** (see 4.2). The dtype and writeback halves are not: `cache_s_l*` is
a ggml-owned tensor and llama.cpp's recurrent-state module picks the f32 dtype for every backend,
so the f16 compute boundary forces a conversion and a `Result` every token.

Ceiling if it were removed: 76.70 -> ~71.5 ms = **~14.0 t/s**, closing the gap from 1.17x to
~1.07x. The remaining `permute_ref` transposes (177 ops, 1.07 ms) and the thin tail of
`RMS`/`StridedSlice`/`Concat`/`Add` ops are small individually and launch-bound - mean non-GEMV op
cost is **4.9 us**, so what matters is op *count*, not op efficiency.

> A first attempt to cut transposes by emitting `Reshape` when `T == 1` (a provable no-op at
> decode, and `Reshape` is `OPTIMIZED_OUT` 1055 times per token while `Transpose` is not) was
> written against `translate_gated_delta_net_ref` - which is **dead code**: the fallback call is
> commented out and the fused `translate_gated_delta_net` is what runs. Op counts were byte-identical
> before and after, confirming it. Reverted. Check which translator actually runs before editing.

### 4.2 Fix landed: GDN state in the op's native layout (+3.0% decode)

`openvino/op/gated_delta_net.cpp` transposed the recurrent state `{0,1,3,2}` on the way into the
fused op and straight back out again, every layer, every token. On the split-outputs path the
state is read from and written back to the same ggml buffer with nothing else observing it, so the
buffer can simply hold the op's layout and both transposes drop out.

OpenVINO's own GGUF frontend made exactly this change upstream (`630516e048`): it declares the
state parameter as `{1, H_v, S, head_v}` and gates the transposes behind a `split_outputs` flag.
We had the matching branch already (`get_output_names().size() == 2`) but still transposed in it.

Guarded on the two state dims being equal - skipping the permute would otherwise also change the
shape ggml allocated. For Bonsai 2 the state is square (128x128), so the guard passes.

| | before | after |
|---|---:|---:|
| `permute_xy_swap` state transposes | 96 ops, 1.73 ms | **0 ops** |
| total `Transpose` | 273 ops, 2.80 ms | 177 ops, 1.08 ms |
| executed ops / token | 2718 | 2622 |
| **tg32 / tg128** | 13.05 / 13.05 | **13.45 / 13.46** |

Prefill unchanged (pp128 236, pp512 535). Output still correct and identical.

**Safety check**: the buffer now holds the op's layout, so every graph touching it must agree.
Instrumented every GDN translation during a `pp128 -n 32,128` run: 144/144 reported
`split_outputs=1 square=1 native=1`, i.e. no graph takes the packed path. If a model ever mixed
the two paths within one session the state would be misread, which is why the change is scoped to
the split-outputs branch rather than applied unconditionally.

### 4.3 Not fixed: the state dtype and writeback (the other ~3.4 ms)

What remains of the state tax after 4.2:

| | n | ms |
|---|---:|---:|
| f32 `Convert` of the ssm state (~48 us each, 3.1 MB/layer) | 48 | ~2.3 |
| `Result` writeback (`cache_r_l*` + `cache_s_l*`) | 96 | 1.06 |

These are one problem, not two: ggml owns `cache_s`/`cache_r` in **f32** while the GPU computes in
f16, so every token crosses that boundary. The `GatedDeltaNet` op itself is not the obstacle - it
validates each input independently against `{f32,f16,bf16}` and sets the output state type from
the *input* state type (`src/core/src/op/gated_delta_net.cpp:103`), so an f32 state alongside f16
q/k/v is legal. The conversion is forced by the plugin's f16 inference precision meeting ggml's
f32 buffer.

The real fix is a device-resident recurrent state (OV `Variable` / `ReadValue`+`Assign`), as
OpenVINO's GGUF frontend does in `pass/make_stateful.cpp`. **`GGML_OPENVINO_STATEFUL_EXECUTION=1`
does not do this** - it covers the KV cache only, leaves the recurrent state round-tripping, and
measured *slower* than stateless here (12.8 vs 13.3 t/s). Implementing it needs a correctness
story for llama.cpp clearing and copying state slots through the cgraph
(`ggml-decoder.h:85`), which an OV-internal state would not observe. Worth ~+4.7% (to ~14.0 t/s).

### sdpa_opt (shape-from-mask) and the opt-in 4-D stateless KV cache

2026-10-16: grafted 15 further `ravi9/dev_backend_openvino` commits (`a4493f9c..287f06326`) onto the
PrismML branch — branch `prism_ov_ternary_kv4d`. Two are the point of the exercise:
`84232f479` (derive attention sizes from the KQ mask shape, so the GPU plugin can pick its fused
SDPA kernel; off-switch `GGML_OPENVINO_DISABLE_SHAPE_FROM_MASK`) and `dd042318d` + `375f0b765`
(4-D stateless KV cache for the plugin's `StatelessKVFusion`, opt-in `GGML_OPENVINO_ENABLE_KV_4D`).

`StatelessKVFusion` needs openvino#37569, which is 2026.5+. The PR #38460 build already **is**
2026.5.0 and already contains #37569, so no toolkit change was needed — which matters, because a
binary nightly would not carry the unmerged TernOCL kernels at all.

Decode, `-n 32,128 -r 5`, each config run on its own (see the measurement note below):

| configuration | tg32 | tg128 |
|---|---:|---:|
| pre-graft (`prism_ov_ternary`) | 12.27 | 12.24 |
| graft, shape-from-mask **off**, KV_4D off | 12.41 | 12.35 |
| graft, shape-from-mask on, KV_4D off (default) | 12.65 | 12.68 |
| graft, shape-from-mask on, **KV_4D on** | **12.84** | **12.86** |
| + `GGML_OPENVINO_DISABLE_KV_SLICE=1` | **13.06** | **13.07** |
| `GGML_OPENVINO_STATEFUL_EXECUTION=1` (llama-cli) | - | 12.8 |

So shape-from-mask is worth ~2.7% and KV_4D ~1.4%, **+5.1% together** over pre-graft. Adding
(This knob was later replaced by `GGML_OPENVINO_KV_SLICE`, default 1; after the 2026-10-30 sync it no longer
affects Bonsai decode - see §4.) At the time, `GGML_OPENVINO_DISABLE_KV_SLICE=1` was worth a further 1.6% with prefill unchanged, for **+6.8%**
total (12.24 -> 13.07) and a 1.15x remaining gap. Stateful execution is worth only +2.4% on this
model and currently **fails llama-bench's generation warmup** (`failed to decode generation batch,
res = -3`) while working fine under llama-cli, so it is not usable for benchmarking here.
Prefill is
unchanged within noise (pp128 232 -> 236, pp512 527 -> 538), so neither change helps the prompt
phase on this model — unsurprising, since prefill here is dominated by the TernOCL int2 GEMM, not
by attention.

Both paths produce correct, identical output (`The capital of France is` -> `Paris`).

These are real but small. They close the gap to the native-OpenVINO ceiling from 1.22x to **1.17x**
(15.03 vs 12.86 at tg128); most of the decode gap is still unexplained by these two optimizations.

Not verified: that the plugin actually selects `sdpa_opt` rather than decomposing to gemm+softmax.
Confirming that needs an `ENABLE_DEBUG_CAPS=ON` OpenVINO rebuild (this build has it off), so the
+2.7% is an observed effect, not a confirmed kernel switch.

> **Measurement note.** `llama-bench -p 128,512 -n 32,128` runs the `pp` cells before the `tg`
> cells and this contaminates `tg32`: in that ordering `tg32` read 11.65 while `tg128` read 12.62
> on the same build, which is backwards. Running `-n 32,128` on its own gives a self-consistent
> 12.65 / 12.68. **The `tg32 12.25 / tg128 11.28` figures in the table above carry this artifact**;
> the clean pre-graft decode number is ~12.25 at both lengths, not 11.28 at tg128. Do not read a
> depth-degradation story into the old numbers.

### The PR is required, not an optimization

With TernOCL disabled (`OV_GPU_ENABLE_TERNOCL_INT2=0`; `OV_TERNOCL_INT2_DISABLE=1` on builds up to `5c6d4692ee`) the u2 weights fall back to OpenVINO's generic decompression
subgraph and the model runs at **1.4 t/s prefill and 1.24 t/s decode** - 167x and 9.9x slower. So
for ternary GGUFs on the OV backend, PR #38460 is the difference between usable and unusable.

Fusing the Hadamard rotation into the FC (the PR's `FuseHadamardIntoFC`, on by default) is worth
about 6% of decode and 1% of prefill once gate/up merge; its real value is that it is the only path
that applies the rotation at all.

## 5. Reproducing

```sh
./bench_bonsai.sh      # both ggml backends, full matrix
```

The native-OpenVINO row needs the u2 IR, which is a one-time build (see
`openvino/src/plugins/intel_gpu/docs/ternocl_int2_bonsai2_from_scratch.md`). It is already built at
`~/llama.cpp/gemma4/ov-bonsai2/`:

```sh
cd ~/llama.cpp/gemma4/ov-bonsai2
TOOLS=~/llama.cpp/gemma4/openvino/src/plugins/intel_gpu/tools/int2
IDS=$(cat $TOOLS/prompt_photosynthesis_27b.txt)
ZE_AFFINITY_MASK=0 BENCH_PRECISION=f16 BENCH_MAX_LEN=512 BENCH_NO_EOS=1 \
  $TOOLS/build/paged_bench_llm_27b bonsai2-27b-u2/openvino_model.xml \
  bonsai2-27b-u2/openvino_text_embeddings_model.xml GPU 128 "$IDS"
```

Three traps that silently produce wrong numbers:

- **llama-bench must warm up.** The OV backend compiles a graph per token count; `--no-warmup`
  puts that one-off compile inside the measured run and pp128 reads 4 t/s instead of 234.
- **Do not use `set -u` around OV `setupvars.sh`** - it reads unset variables, kills the subshell,
  and you get empty output with no error.
- **`test-backend-ops` device/backend must agree.** With `GGML_OPENVINO_DEVICE=CPU` the backend is
  `OPENVINO0`; passing `-b OPENVINO1` runs zero tests and still prints `0/0 tests passed` and an
  overall OK.

## 6. Open items

- Tile tables are Xe2-tuned; this GPU is Xe3, so prefill numbers are a floor, not a ceiling.
- **Decode gap to GenAI, 1.10x** (on OpenVINO master, after the sync). Root-caused in §4.1 (non-GEMV graph overhead; the GPU is
  saturated on both paths). The largest remaining item is the f32 recurrent-state boundary
  (§4.3), which needs a device-resident state rather than a local change.
- **The newer PR / OpenVINO master costs ~1.5% of ggml-openvino decode** vs PR revision
  `5c6d4692ee` (13.47 -> 13.28). Not bisected.
- Int8 prefill looks like it should be the default for this model, but that is the PR author's
  call and needs a GSM8K/WWB-style accuracy check, not a 90-token diff.
- ggml-openvino has no PTQ1_0 support (Bonsai 1).
- The 5 pre-existing `GATED_DELTA_NET` op failures on `dev_backend_openvino` are unrelated to this
  work but affect `raw_gates` / `rows_mode` paths this arch family can use.
- Pushed to `cavusmustafa/llama.cpp` branch `prism_ov_ternary_sync` (synced; supersedes `prism_ov_ternary_kv4d`); no PR is open.
