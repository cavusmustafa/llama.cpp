#!/bin/bash
# Bonsai 2 27B: effect of the grafted sdpa_opt (shape-from-mask) and the opt-in
# 4-D stateless KV cache (StatelessKVFusion) on the TernOCL OV build.
# Warmup must stay ENABLED (see bench_bonsai.sh).
set +u
ROOT=/home/mcavus/llama.cpp/gemma4
MODEL=$ROOT/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf
ARGS="-p 128,512 -n 32,128 -r 3 -ngl 99"
source "$ROOT/openvino/install-pr38460/setupvars.sh" >/dev/null 2>&1
cd "$ROOT/bonsai-ov" || exit 1

echo "######## A: default (KV_4D off, shape-from-mask ON) ########"
GGML_OPENVINO_DEVICE=GPU ./build-pr/bin/llama-bench -m "$MODEL" $ARGS

echo; echo "######## B: GGML_OPENVINO_ENABLE_KV_4D=1 ########"
GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_ENABLE_KV_4D=1 ./build-pr/bin/llama-bench -m "$MODEL" $ARGS

echo; echo "######## C: KV_4D=1 + DISABLE_SHAPE_FROM_MASK=1 (isolate sdpa_opt) ########"
GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_ENABLE_KV_4D=1 GGML_OPENVINO_DISABLE_SHAPE_FROM_MASK=1 \
  ./build-pr/bin/llama-bench -m "$MODEL" $ARGS
