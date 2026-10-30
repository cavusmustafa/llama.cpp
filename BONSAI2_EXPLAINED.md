# Running a ternary 27B model on an Intel GPU - how it all fits together

A walkthrough of what Bonsai 2 27B is, how its 2-bit weights are stored, why it needs a
Hadamard rotation at run time, what had to change in llama.cpp's OpenVINO backend to make it run
on an Intel GPU, and where its decode time actually goes.

Written to be readable without prior knowledge of this codebase. Measured numbers and the terse
engineering version are in `BONSAI2_TERNARY_OV_REPORT.md`; this file explains the *why*.

Last updated 2026-10-17: sections 8 and 9 are new or rewritten.

---

## TL;DR

| | |
|---|---|
| **Model** | Bonsai 2 27B, a ternary (-1 / 0 / +1) quantization of a 27 B hybrid-attention LLM. 6.70 GiB at 2.13 bits per weight, vs ~54 GB at fp16. |
| **Goal** | Run it through llama.cpp's OpenVINO backend on an Intel GPU, using the new 2-bit OpenCL kernels from OpenVINO PR #38460, and compare against Vulkan and against native OpenVINO. |
| **Work needed** | 161 lines across 7 files to make it run, then four more changes to close part of the decode gap. |
| **Outcome** | Correct output, **3.21x faster prefill than Vulkan** (544 vs 169 t/s), and decode improved 12.24 -> 13.46 t/s. On the latest PR build, real **OpenVINO GenAI** is 1.10x ahead on decode and 1.21x on prefill. |
| **Biggest surprise** | Without the PR's 2-bit kernels, the same model runs **167x slower**. The kernels are not an optimization, they are the difference between usable and unusable. |
| **Correction** | This page previously said decode was a tie limited by an ~80 GB/s bandwidth wall. Both halves were wrong - the peak is 153.6 GB/s and the limit was integration overhead. See section 9. |

---

## 1. What the model is

![architecture](diagrams_bonsai2/01-architecture.svg)

Bonsai 2 27B declares itself as architecture `qwen35` in its GGUF file - a 27 B dense model
(no mixture-of-experts), 64 layers, hidden size 5120, 248320-token vocabulary, 262 k context.

The interesting part is that it is a **hybrid attention** model. Only every 4th layer uses the
attention you would expect. The other three quarters use a *linear attention* mechanism called
**GatedDeltaNet**:

- **48 layers - GatedDeltaNet.** Instead of comparing every token against every other token,
  this keeps a fixed-size recurrent state (a 128 x 128 matrix per head) and updates it as tokens
  stream past. Cost per token is constant, so long contexts stay cheap. This is why the model can
  claim a 262 k context without a 262 k-sized attention matrix.
- **16 layers - full attention.** Normal softmax attention with 24 query heads over 4 key/value
  heads, so the model still has a way to do precise long-range lookups.

Every layer, of both kinds, then runs the same SwiGLU feed-forward network
(5120 -> 17408 -> 5120).

### Why that matters here

The GatedDeltaNet layers are where two of the four fixes landed. Linear attention has a more
intricate graph than plain attention - a 1-D convolution over q/k/v, per-head normalization,
two learned gates, and a recurrent scan - and that gave more surface area for shape bugs.

---

## 2. How PQ2_0 stores a ternary weight

![pq2_0 block](diagrams_bonsai2/02-pq2_0-block.svg)

Every weight in the big projections is one of three values: `-1`, `0`, `+1`. Storing a *trit*
needs `log2(3) = 1.585` bits in theory. PQ2_0 ("Prism Q2_0", a ggml type private to the PrismML
fork) spends a bit more for simplicity:

- Weights are grouped in **blocks of 128**.
- Each block has **one fp16 scale** `d` (2 bytes) and **32 bytes of codes** (128 codes x 2 bits).
- A code is `0`, `1` or `2`, and the value is **`(code - 1) * d`**. Code `3` is never used.

That is 34 bytes per 128 weights = **2.125 bits per weight**. The 2 bits of payload plus
`16 / 128 = 0.125` bits of scale is exactly where the number comes from.

The wasted 4th code buys something real: unpacking is a shift and a mask, nothing more. The
sibling format `PTQ1_0` packs 5 trits into a byte in base 3 and reaches 1.75 bpw, but then
unpacking needs divisions. Prism ships both and lets you choose.

> **A useful way to think about it:** the model is not "compressed and then decompressed". The
> weights genuinely *are* -1/0/+1. The scale just restores the magnitude per group of 128.

---

## 3. The twist: the weights live in a rotated basis

![hadamard](diagrams_bonsai2/04-hadamard.svg)

This is the part that makes the model unusual to support, and it is worth understanding before
looking at any code.

**The problem.** Ternary quantization has three levels per group. If a group of 128 numbers is
mostly small but contains one large outlier, the scale gets set by the outlier and everything
else rounds to 0. You lose almost all the information.

**The trick.** Before quantizing, multiply the weight's input axis by a **Hadamard matrix** - an
orthogonal matrix of `+s` and `-s` entries. Every output channel becomes a plus/minus sum of all
1024 input channels, so any single outlier is smeared across the whole block and the values turn
roughly Gaussian. Ternary then loses far less.

Because the matrix is orthogonal, the math is unchanged *provided you rotate the activation the
same way*:

```
W_rot = W . H        (done once, offline, by whoever quantized the model)
x_rot = H . x        (must happen at run time, every forward pass)

W_rot . x_rot  ==  W . x
```

**The consequence.** If you load these weights and skip the activation rotation, you do not get a
slightly worse answer - you get garbage. The model card warns about exactly this. So the rotation
is not an optimization detail, it is part of the model definition.

For this model the transform is a 1024-wide normalized Sylvester-Walsh-Hadamard matrix (scaled by
`1/sqrt(1024) = 1/32`), preceded by a fixed per-channel sign flip. llama.cpp does not load the
matrix from the file - it *generates* it at load time from a parity rule, and reads the sign
vectors out of the GGUF metadata.

In the ggml graph this shows up as four ordinary ops in front of every ternary matmul:

```
x -> MUL by signs -> RESHAPE to [T*K/1024, 1024] -> MUL_MAT by H -> RESHAPE back -> MUL_MAT by W
```

Nothing exotic. Which is good news: a backend that supports `MUL`, `RESHAPE` and `MUL_MAT`
already supports the rotation - just slowly, since it is ~400 extra 1024x1024 matmuls per token.
The OpenVINO GPU plugin has a pass that recognises that chain and folds it into the matmul, so a
fast fused kernel does it instead.

---

## 4. Teaching the OpenVINO backend to read PQ2_0

![pq2_0 to u2](diagrams_bonsai2/03-pq2_0-to-u2.svg)

OpenVINO has no *signed* 2-bit type. It has unsigned `u2`. So ternary is represented as codes
`{0, 1, 2}` with a **zero point of 1**, dequantized as `(code - 1) * scale`.

If that sounds familiar, it is because it is byte-for-byte what PQ2_0 already does. And the bit
packing matches too: both put element `j` at bits `[2*(j%4), +1]` of byte `j/4`, lowest bits
first. (OpenVINO packs every sub-byte type LSB-first except `u1`.)

**So the conversion is a `memcpy`.** The 32 code bytes of each block are copied straight into an
OpenVINO `u2` constant, with no bit shuffling and no re-encoding. Only the fp16 scales need
gathering into a separate array, because ggml interleaves scale and codes while OpenVINO wants
them separate.

The backend then emits the standard "weight decompression" shape that the GPU plugin knows how to
fold into a compressed matmul:

```
Constant(u2) -> Convert(f16) -> Subtract(1) -> Multiply(scales) -> Reshape -> MatMul
```

Two things here are easy to get wrong, and both were real bugs:

1. **The zero point must be a `Constant`, not a `Convert` of an integer constant.** The ternary
   kernel reads that value directly out of memory when it builds. A `Convert` node has not
   produced a value yet, so the fusion silently does not happen.
2. **Never requantize a ternary tensor.** The backend had a rule that forced the token embedding
   and output projection to 8-bit, for good reasons on other models. On this model it inflated
   2.13 bpw to 8 bpw *and* knocked the output projection off the ternary path. That is fix 4.

---

## 5. The four fixes

![pipeline](diagrams_bonsai2/05-pipeline.svg)

The diagram above shows the whole path from GGUF file to GPU kernel, with the four changes
marked. Two of them are about the ternary format; two are about shapes.

### Fix 1 - read PQ2_0 as `u2`

Covered in section 4. Mechanical once you notice the bit layouts already agree.

### Fix 2 - the joint q/k view in GatedDeltaNet

![gdn fix](diagrams_bonsai2/06-fix-gdn.svg)

This model normalizes its query and key tensors **together**, as one 32-head tensor, then takes
two 16-head views of the result. That is cheaper than two separate normalizations.

The OpenVINO backend has a design rule: translating a `VIEW` is a **no-op pass-through**, and the
*consumer* is responsible for slicing out the part it wants. The GatedDeltaNet translator honoured
that rule for its `v` input - and only for `v`. So `q` and `k` each arrived carrying all 32 heads.

Then it got worse. Linear attention here is grouped-query: 16 q/k heads serve 48 value heads, so
q and k get tiled 3x. The tiling factor was computed from the *declared* shape (16, correct) but
applied to the *actual* tensor (32, wrong), producing 96 heads against 48 value heads. The op
rejected it.

The fix re-slices `q` and `k` on the head axis, exactly as `v` already was. It is a no-op when the
declared width already matches the actual width, so models that normalize q and k separately -
including Qwen3.5 and Qwen3.6 35B-A3B, which is why nobody had hit this - are untouched.

**Worth internalising:** the declared shape said 16 and the real tensor was 32. If you only look
at the metadata, this bug is invisible. You have to compare declared against actual.

### Fix 3 - a reshape that baked in the token count

![reshape fix](diagrams_bonsai2/07-fix-reshape.svg)

The OpenVINO backend compiles a model whose **token dimension is dynamic**, then replays that one
compiled model for a 42-token prompt, a 2-token warmup, a 1-token decode step, and so on. So no
shape written into the graph as a constant may depend on the token count `T`.

The backend works out which output axis of a reshape is "the dynamic one" so it can write `-1`
there instead of a number. Its rule was: find the axis with the same stride *and the same extent*
as the source's dynamic axis.

The Hadamard rotation breaks that rule, because its reshape **splits** the token axis:
`[T, 5120]` becomes `[T*5, 1024]`. No output axis has extent `T` any more - it has extent `T*5`.
So the rule found nothing, the whole target shape was written as a constant, and the number baked
in came from whichever graph was captured first (a 2-token warmup, giving `[1,1,10,1024]`).
Replaying with 42 tokens then threw a shape error.

The fix adds a fallback that runs only when the old rule fails: find the output axis that *spans
the same elements* - everything above it static, and the same element count up to and including
it. This is fully general; nothing about it is ternary- or Bonsai-specific, and it closes a latent
class of bugs affecting any reshape that splits or merges the dynamic axis.

### Fix 4 - keep ternary weights out of requantization

The backend requantized `token_embd.weight` and `output.weight` to 8-bit unconditionally. Every
requantization target is 4-bit or wider, so on a ternary model this can only make things worse.
Return "no requantization" for PQ2_0 and move on.

---

## 6. The debugging lesson

![fix chain](diagrams_bonsai2/08-fix-chain.svg)

This is the most transferable thing in the whole project, so it gets its own diagram.

Fix 3 (the baked reshape) did not announce itself as a shape bug. It surfaced, four steps later,
as this:

```
[GPU] ternocl int2: ...ffn_gate... carries a Hadamard input transform
      but the TernOCL impl rejected it: fused chain has no folded epilogue
```

Nothing in that message is wrong, and nothing it names is broken. The causal chain was:

1. The reshape target was a constant, so the shape downstream became **static**.
2. `FullyConnectedHorizontalFusion` - the pass that merges `ffn_gate` and `ffn_up` into one
   matmul - starts with `if (!input_shape.is_dynamic()) return false;`. It silently bailed.
3. So `ffn_gate` stayed a separate node and accumulated **three** fused element-wise ops
   (the SiLU, the multiply by `up`, and the next layer's Hadamard sign multiply).
4. The ternary kernel's epilogue table only knows chains of length 1 or 2. It rejected the node.
5. Normally a rejection is harmless - another implementation takes over. But this matmul had its
   Hadamard rotation folded in, and the ternary kernel is the *only* implementation that applies
   that rotation. Falling back would silently produce garbage, so the plugin **escalates the
   rejection to a hard build failure** on purpose.

Chasing the error message leads into the kernel's acceptance rules, five steps from the cause.
Fixing the reshape made the merge fire and the entire cascade vanish at once.

> **Rule of thumb:** when a GPU-plugin fusion mysteriously does not fire, check whether the shapes
> are still dynamic *before* studying the pattern matcher. Several passes silently require dynamic
> shapes, because they only target generative models.

The confirmation that the merge was the missing piece: after fix 3, the kernel's debug output
names the accepted nodes `ffn_gate_fused_2FCs`, `z_fused_2FCs`, `Qcur_full_fused_3FCs` - the
`_fused_NFCs` suffix is the horizontal fusion naming its merged result.

---

## 7. How one compiled model uses two different kernels

![kernel dispatch](diagrams_bonsai2/09-kernel-dispatch.svg)

A natural question: if prefill and decode run on the same compiled model, how can they use
different kernels?

They can because **the kernel is not chosen at compile time**. Inside the matmul primitive, on
every inference call:

```cpp
const size_t M = ov::shape_size(params->output_layouts[0].get_shape()) / _N;  // tokens, at run time
if (M > 8 && int8_prefill_enabled())
    return execute_int8(...);
auto& l = get_launch(M);
```

`launch_class(M)` buckets the token count into 8 classes. `get_launch` returns the cached kernel
for that class, or builds it with `clBuildProgram` on first use and keeps it in an 8-slot array.
So a single matmul node accumulates a handful of OpenCL kernels over the process lifetime and
picks among them per call:

- **M <= 8** (decode): a GEMV, tuned for one or a few tokens.
- **M > 8** (prefill): a tiled GEMM.

Both read the *same* packed weights, but move the work around differently:

| | weights | activations |
|---|---|---|
| **int2 x f16 up-convert** (default) | dequantized to fp16 in registers, then fp16 matrix instructions | stay in native fp16 - no extra error |
| **int2 x int8** (`OV_GPU_TERNOCL_INT2_INT8_PREFILL=1`, prefill only) | never unpacked - raw 2-bit codes feed a native int8 x int2 matrix instruction | quantized to int8 per (row, 128-group) - faster, slightly lossy |

### What `OV_GPU_TERNOCL_INT2_INT8_PREFILL=1` actually does

> **The name depends on the OpenVINO build.** Current PR #38460 heads expose it as the plugin
> property `GPU_TERNOCL_INT2_INT8_PREFILL`, set from the environment as
> **`OV_GPU_TERNOCL_INT2_INT8_PREFILL=1`**. Builds up to `5c6d4692ee` read a plain environment
> variable, `OV_TERNOCL_INT2_INT8_PREFILL=1`, instead. Each build **silently ignores the other
> name**: with the old spelling on a current build, pp128 stays at the default 237 t/s with no
> warning. If the int8 numbers do not move, check the spelling first.

It is **off by default**, and it changes only how the *prompt* is processed - the weights on disk
are untouched and decode is unaffected.

By default the ternary kernel **up-converts**: it unpacks each 2-bit code to fp16 in registers and
runs fp16 matrix instructions against fp16 activations. Correct, lossless on the activation side,
but it spends the GPU's fp16 matrix throughput on operands that only ever hold three distinct
values.

With the switch on, prefill instead takes the **int2 x int8 DPAS** path:

- the weights are **never unpacked** - the raw 2-bit codes feed the matrix instruction directly;
- the **activations** are quantized to int8, one scale per (row, 128-group);
- the hardware's integer matrix instruction runs at a higher rate than the fp16 one.

It only engages where it pays off - `M > 8`, i.e. a prompt or a batch, never the `M = 1` decode
step (see the dispatch snippet above). That is why every int8 row in the results is a prefill
number and the decode column is blank.

**The trade is accuracy for prompt speed.** Quantizing activations to int8 is lossy in a way the
default path is not. On this model it was harmless - default and int8 prefill were
byte-identical over 90 greedy tokens here, and the PR author reports GSM8K within the standard
error of the default. It is still the reason the switch is opt-in rather than the default: that is
the PR author's call to make, and it needs a real accuracy harness rather than a 90-token diff.

Measured here: **pp128 236 -> 546 t/s (2.3x)** and **pp512 544 -> 575 t/s (1.06x)**. The gain is
much larger at short prompts because at pp512 the default path is already close to saturating the
machine, so there is less headroom left to recover.

---

## 8. Results

![results](diagrams_bonsai2/11-results-2026-10-17.svg)

Intel Arc B390 iGPU, same GPU and same model file for every row, all on the **latest PR #38460
head** (`4185ce2f15`). Decode measured with `-n` alone and prefill with `-p` - see the measurement
note below, which is not a formality.

| | pp128 | pp512 | tg32 | tg128 |
|---|---:|---:|---:|---:|
| **OpenVINO GenAI** (`VLMPipeline`, `llm_bench`) | 256.2 | **657.9** | **14.63** | **14.53** |
| **OpenVINO GenAI**, int8 prefill | **638.2** | **715.1** | - | - |
| **ggml-openvino** + ternary kernels | 236.5 | 544.2 | 13.21 | 13.21 |
| **ggml-openvino**, int8 prefill | 545.8 | 575.1 | - | - |
| **ggml-vulkan** | 171.2 | 169.5 | 12.70 | 12.70 |
| ggml-openvino, ternary kernels off | 1.4 | - | 1.24 | - |

"int8 prefill" is `OV_GPU_TERNOCL_INT2_INT8_PREFILL=1` (section 7). GenAI prefill t/s is prompt
tokens / first-token latency at an exact 128- or 512-token prompt; GenAI decode is llm_bench's
2nd-token throughput.

GenAI / ggml-openvino: pp128 **1.08x**, pp512 **1.21x**, decode **1.10-1.11x** (int8 prefill on
both: 1.17x / 1.24x). ggml-openvino / Vulkan: pp512 **3.21x** (3.39x with int8 prefill), decode
**1.04x**.

<details markdown="1"><summary>The same comparison on the earlier PR build (<code>5c6d4692ee</code>)</summary>

The earlier PR revision did not ship a GenAI path; its native-OpenVINO reference was the PR's own
`paged_bench_llm_27b` PagedAttention tool, since removed from the PR. That build decodes about 2%
faster under ggml-openvino (13.47 vs 13.21 t/s, measured back to back).

| | pp128 | pp512 | tg32 | tg128 |
|---|---:|---:|---:|---:|
| native OpenVINO (`paged_bench_llm_27b`) | 241.2 | 638.5 | 14.90 | 15.03 |
| native OpenVINO, int8 prefill | 616.3 | 693.4 | - | - |
| ggml-openvino + ternary kernels | 236.1 | 535.1 | 13.45 | 13.46 |
| ggml-openvino, int8 prefill | 549.0 | 579.7 | - | - |
| ggml-openvino, before the 2026-10-17 work | 232.2 | 527.3 | 12.27 | 12.24 |

</details>

**Prefill: OpenVINO wins by 3.16x** at a 512-token prompt (3.42x with int8 prefill). The shape of
the curve is as informative as the ratio - Vulkan is flat from 128 to 512 tokens (171 -> 169 t/s),
which is what a per-token kernel does, while OpenVINO scales (236 -> 535), which is what a tiled
GEMM does.

**Decode is not a tie.** An earlier version of this page said it was, and that both backends were
pinned to a ~80 GB/s hardware wall. That was wrong twice over. This machine's memory is
8 x 16-bit LPDDR5X-9600 = **153.6 GB/s** peak, so ~80 GB/s was never the ceiling; and native
OpenVINO sustains **15.03 t/s** on the same GPU and the same weights, which no bandwidth wall at
12 t/s would allow. The wall was an *integration* ceiling, not a hardware one. Section 9 is about
finding and partly removing it.

**The PR's kernels are mandatory.** Turn them off and the same model drops to 1.4 t/s prefill.
(Caveat: that switch also disables the gate/up merge and the Hadamard fold, so it measures "whole
PR off vs on", not the kernel alone.)

> **Measuring decode correctly.** `llama-bench -p 128,512 -n 32,128` runs the prefill cells first
> and contaminates the first decode cell - in that ordering tg32 read 11.65 while tg128 read 12.62
> on the same build, which is backwards. Run `-n <N>` on its own. The `tg128 11.28` figure in the
> old table above was this artifact, and it is what produced a now-retracted claim that decode
> degrades with context depth. Measured properly, decode is flat.

### Correctness

- OpenVINO GPU output is **identical to the CPU reference** on a short greedy prompt.
- Default and `INT8_PREFILL=1` were **byte-identical** over 90 greedy tokens here.
- Vulkan agrees in substance but diverges in wording after ~35 greedy tokens - ordinary near-tie
  drift between different kernels, not a correctness failure.
- The op-level test suite: **8079 / 8084**, and all 5 failures reproduce on an unmodified build,
  so they pre-date this work.

---

## 9. Closing the decode gap

Everything above gets the model running correctly and fast at prefill. This section is the
follow-up question: native OpenVINO decodes the same weights on the same GPU at 15.03 t/s and we
were at 12.24. **Where does the difference go?**

### First, rule out the two usual suspects

![decode budget](diagrams_bonsai2/13-decode-budget.svg)

The two standard explanations are "we are bandwidth-bound" and "we are dispatch-bound" (the host
cannot feed the GPU fast enough). Both are measurable, and both turned out to be wrong:

- **GPU occupancy**: 99.99% busy for ggml-openvino, 99.95% for native OpenVINO, measured off the
  `xe` PMU during steady-state decode. The GPU is saturated on both paths, so this is not host
  dispatch and not idle time.
- **The ternary GEMVs are already fast**: 6.86 GB of weights in 65.7 ms is **106 GB/s**, 69% of
  the 153.6 GB/s peak. And both backends run the *same* TernOCL kernels over the *same* u2
  weights, so that term is identical on both sides and cannot be the difference.

What is left is the scaffolding around the GEMVs: `64.7 GEMV + 2.5 overhead` for native OpenVINO
versus `65.7 + 10.5` for ggml-openvino. **The entire gap is ~10 ms/token of non-GEMV graph ops.**
The mean non-GEMV op costs 4.9 us, so what matters is how *many* ops there are, not how fast each
one is.

### The biggest single item: the recurrent state was being shipped back and forth

![gdn state round-trip](diagrams_bonsai2/12-gdn-state-roundtrip.svg)

Grouping the overhead by *cause* instead of by op type made one mechanism jump out. Three
different op types each appeared exactly **96 times = 48 GDN layers x 2**:

| per-token cost | op | n | ms |
|---|---|---:|---:|
| state layout swap `{0,1,3,2}` | `Transpose` | 96 | 1.73 |
| state dtype conversion to f32 | `Convert` | 96 | 2.43 |
| state writeback | `Result` (`cache_s_l*`, `cache_r_l*`) | 96 | 1.06 |

All of it is the cost of handing the GDN recurrent state across the graph boundary and back, every
token. The state is `[1, 48, 128, 128]` f32 - 3.1 MB per layer, ~151 MB per token in total.

The **layout** half is now fixed. `translate_gated_delta_net` was transposing the state into the
fused op's `[B, H, key, value]` on the way in and transposing it straight back to ggml's
`[B, H, value, key]` on the way out. But on this path the state is read from and written back to
the same ggml buffer with nothing else looking at it in between - so the buffer can simply hold
the op's layout and both transposes disappear. Guarded on the two state dims being equal, since
otherwise skipping the permute would also change the shape ggml allocated (Bonsai 2's state is
square, 128x128). OpenVINO's own GGUF frontend made the identical change upstream in `630516e048`.

Worth 96 ops and 1.73 ms/token: **13.05 -> 13.45 t/s**, prefill unchanged, output identical.

### What that adds up to, and what is left

Four changes landed, three of them ported from `ravi9/dev_backend_openvino` and one written here:

| change | decode |
|---|---|
| attention sizes derived from the KQ-mask shape (so the GPU picks its fused SDPA kernel) | +2.7% |
| 4-D stateless KV cache, `GGML_OPENVINO_ENABLE_KV_4D=1` (needs OpenVINO >= 2026.5) | +1.4% |
| `GGML_OPENVINO_DISABLE_KV_SLICE=1` | +1.6% |
| GDN recurrent state kept in the op's layout | +3.0% |
| **total** | **12.24 -> 13.46 t/s, +10.0%** |

The gap to native OpenVINO is now **1.12x**, down from 1.22x.

The remaining ~3.4 ms/token is the **dtype and writeback** halves of the same state round-trip, and
they are one problem rather than two: ggml owns the recurrent state in **f32** while the GPU
computes in f16, so the state crosses that boundary every token. The fused op is not the obstacle -
it type-checks each input independently against `{f32, f16, bf16}` and takes its output state type
from its *input* state type, so an f32 state beside f16 activations is perfectly legal. The fix is
to keep the state resident on the device as an OpenVINO `Variable`, which is what OpenVINO's GGUF
frontend does in its `make_stateful` pass. Two cautions for whoever picks that up:
`GGML_OPENVINO_STATEFUL_EXECUTION=1` does **not** do this today (it covers the KV cache only, and
measured *slower* here), and the hard part is not the plumbing but staying correct when llama.cpp
clears and copies state slots through the compute graph - operations an OpenVINO-internal state
would never see.

---

## 10. What you need to run it

All four pieces are required, and each fails differently if missing:

| piece | if missing |
|---|---|
| `PrismML-Eng/llama.cpp` (branch `prism`) | the file will not load - stock llama.cpp rejects the type |
| the four ggml-openvino fixes | crash at graph build, or wrong output |
| OpenVINO built with PR #38460 | runs, but 167x slower |
| an Intel GPU with the OpenCL runtime | the ternary kernels require it |

```sh
source <openvino-install>/setupvars.sh
GGML_OPENVINO_DEVICE=GPU ./build-pr/bin/llama-cli \
  -m Ternary-Bonsai-2-27B-PQ2_0.gguf -p "The capital of France is" \
  -n 20 -c 512 --no-warmup -st --temp 0
```

Useful switches while poking at it:

| variable | effect |
|---|---|
| `OV_GPU_TERNOCL_INT2_INT8_PREFILL=1` | the faster, slightly lossy prefill kernel (old builds: `OV_TERNOCL_INT2_INT8_PREFILL=1`) |
| `OV_GPU_ENABLE_TERNOCL_INT2=0` | turn the TernOCL kernels off (old builds: `OV_TERNOCL_INT2_DISABLE=1`) |
| `OV_GPU_ENABLE_HADAMARD_FC_FUSION=0` | leave the rotation in the graph instead of folding it - debug-caps builds only (old builds: `OV_TERNOCL_INT2_FUSE_HADAMARD=0`) |
| `OV_TERNOCL_INT2_DEBUG=1`, `_CFG_DEBUG=1`, `OV_TERNOCL_HADAMARD_DEBUG=1` | accept/reject, tile choice and rotation tracing - **old builds only**; current heads route this through debug-caps logging |
| `GGML_OPENVINO_DUMP_CGRAPH=1` | dump the ggml graph to `cgraph_ov.txt` |

**One benchmarking trap worth repeating:** the OpenVINO backend compiles a graph per token count,
and that compile takes tens of seconds on a 27 B model. `llama-bench --no-warmup` puts it *inside*
the measured run and reports 4 t/s instead of 234. Always let it warm up.
