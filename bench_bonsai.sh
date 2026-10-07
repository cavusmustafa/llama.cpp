#!/bin/bash
# Bonsai 2 27B PQ2_0: ggml-openvino (TernOCL int2) vs ggml-vulkan on the same GPU.
#
# Warmup must stay ENABLED. The OpenVINO backend compiles a graph per token count,
# and --no-warmup puts that one-off compile (tens of seconds on a 27B) inside the
# measured run: pp128 reads 4 t/s instead of 233 t/s.
set +u
ROOT=/home/mcavus/llama.cpp/gemma4
MODEL=$ROOT/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf
ARGS="-p 128,512 -n 32,128 -r 3 -ngl 99"

cd "$ROOT/bonsai-ov" || exit 1

echo "############ ggml-openvino (OV GPU, TernOCL int2) ############"
( source "$ROOT/openvino/install-pr38460/setupvars.sh" >/dev/null 2>&1
  GGML_OPENVINO_DEVICE=GPU ./build-pr/bin/llama-bench -m "$MODEL" $ARGS )

echo
echo "############ ggml-vulkan ############"
( source "$ROOT/1.4.350.1/setup-env.sh" >/dev/null 2>&1
  ./build-vk/bin/llama-bench -m "$MODEL" $ARGS )
