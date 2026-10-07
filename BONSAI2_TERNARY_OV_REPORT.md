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
- OV GPU default and `OV_TERNOCL_INT2_INT8_PREFILL=1` are **byte-identical** over 90 greedy tokens.
- Vulkan agrees semantically but diverges in wording after ~35 greedy tokens. This is ordinary
  near-tie drift between different kernels, not a correctness failure.
- `test-backend-ops -b OPENVINO0`: **8079/8084**. All 5 failures are `GATED_DELTA_NET`
  (`raw_gates=1` / `rows_mode=1`) and reproduce identically on the pristine pre-change graft build,
  so they are pre-existing on `dev_backend_openvino`.
- llama-3.2-1B and gemma-4-E2B still produce correct output at unchanged speed on OV GPU, so the
  global reshape change did not regress other models.

## 4. Performance

`llama-bench -r 3 -ngl 99`, same GPU, same model file. t/s.

| configuration | pp128 | pp512 | tg32 | tg128 |
|---|---:|---:|---:|---:|
| **ggml-openvino** + TernOCL (default) | 233.9 | 527.4 | 12.25 | 11.28 |
| **ggml-openvino** + TernOCL, `INT8_PREFILL=1` | **530.2** | **571.1** | 11.64 | - |
| **ggml-vulkan** | 161.0 | 165.4 | **12.32** | **12.31** |
| ggml-openvino, `TERNOCL_DISABLE=1` | 1.4 | - | 1.24 | - |

### Prefill

ggml-openvino is **3.2x ggml-vulkan at pp512** (527 vs 165), and **3.5x** with the optional int8
prefill kernel (571). Vulkan is essentially flat from pp128 to pp512 (161 -> 165) while OV scales
(234 -> 527), which is what a tiled DPAS GEMM should do against a per-token kernel.

`OV_TERNOCL_INT2_INT8_PREFILL=1` is off by default in the PR but is a large win here, 2.3x at
pp128, and was bit-identical on this model.

### Decode

A tie, and both backends are **memory-bandwidth-bound, not kernel-bound**: 6.70 GiB x ~12 t/s is
about 80 GB/s, which is roughly all this iGPU can sustain. There is no headroom to win here, and
decode on this model cannot distinguish the two backends on this hardware. A discrete part is
needed to say anything useful about decode.

### The PR is required, not an optimization

With `OV_TERNOCL_INT2_DISABLE=1` the u2 weights fall back to OpenVINO's generic decompression
subgraph and the model runs at **1.4 t/s prefill and 1.24 t/s decode** - 167x and 9.9x slower. So
for ternary GGUFs on the OV backend, PR #38460 is the difference between usable and unusable.

Fusing the Hadamard rotation into the FC (the PR's `FuseHadamardIntoFC`, on by default) is worth
about 6% of decode and 1% of prefill once gate/up merge; its real value is that it is the only path
that applies the rotation at all.

## 5. Reproducing

```sh
./bench_bonsai.sh      # both backends, full matrix
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
- `OV_TERNOCL_INT2_INT8_PREFILL` looks like it should be the default for this model, but that is the
  PR author's call and needs a GSM8K-style accuracy check, not a 90-token diff.
- The 5 pre-existing `GATED_DELTA_NET` op failures on `dev_backend_openvino` are unrelated to this
  work but affect `raw_gates` / `rows_mode` paths this arch family can use.
- Nothing here has been pushed or proposed upstream.
