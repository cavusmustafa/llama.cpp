#!/bin/bash
# Bonsai 2 27B on OpenVINO master (5d07300df3, PR #38460 merged): synced vs pre-sync
# ggml-openvino, plus OpenVINO GenAI. Decode measured with -n alone (pp-before-tg
# contaminates the first tg cell).
set +u
ROOT=/home/mcavus/llama.cpp/gemma4
M=$ROOT/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf
source $ROOT/openvino/install-master/setupvars.sh >/dev/null 2>&1
f(){ sed 's/.*| *\([pt][pg][0-9]*\) *| *\([0-9.]*\) ± \([0-9.]*\).*/    \1: \2 ± \3/'; }
lb(){ # label, binary-dir, env...
  local lbl=$1 dir=$2; shift 2
  echo "## $lbl"
  env GGML_OPENVINO_DEVICE=GPU "$@" $dir/build-master/bin/llama-bench -m $M -p 128,512 -r 3 -ngl 99 2>/dev/null | grep -E "pp(128|512)" | f
  for n in 32 128; do env GGML_OPENVINO_DEVICE=GPU "$@" $dir/build-master/bin/llama-bench -m $M -n $n -r 5 -ngl 99 2>/dev/null | grep "tg$n" | f; done
}
SYNC=$ROOT/bonsai-ov; OLD=$ROOT/bonsai-kv4d
lb "synced, defaults"                         $SYNC
lb "synced, KV_4D=1 KV_SLICE=0"               $SYNC GGML_OPENVINO_ENABLE_KV_4D=1 GGML_OPENVINO_KV_SLICE=0
lb "synced, KV_4D=1 (KV_SLICE default on)"    $SYNC GGML_OPENVINO_ENABLE_KV_4D=1
lb "pre-sync, KV_4D=1 DISABLE_KV_SLICE=1"     $OLD  GGML_OPENVINO_ENABLE_KV_4D=1 GGML_OPENVINO_DISABLE_KV_SLICE=1
echo "## int8 prefill (OV_GPU_TERNOCL_INT2_INT8_PREFILL=1)"
echo "  synced:";   env GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_ENABLE_KV_4D=1 GGML_OPENVINO_KV_SLICE=0 OV_GPU_TERNOCL_INT2_INT8_PREFILL=1 $SYNC/build-master/bin/llama-bench -m $M -p 128,512 -r 3 -ngl 99 2>/dev/null | grep pp | f
echo "  pre-sync:"; env GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_ENABLE_KV_4D=1 GGML_OPENVINO_DISABLE_KV_SLICE=1 OV_GPU_TERNOCL_INT2_INT8_PREFILL=1 $OLD/build-master/bin/llama-bench -m $M -p 128,512 -r 3 -ngl 99 2>/dev/null | grep pp | f
echo "## OpenVINO GenAI (master)"
W=$ROOT/ov-bonsai2; cd $W/openvino.genai/tools/llm_bench
g(){ env $2 $W/bench-venv/bin/python benchmark.py -m $W/bonsai2-27b-genai -d GPU -pf $3 --disable_prompt_permutation -ic $4 -n 3 2>&1 | grep "\[Average\]" | sed "s/^.*P\[0\]/    $1:/"; }
g "pp128 default" ""                                   $W/genai_prompts/p128.jsonl 8
g "pp512 default" ""                                   $W/genai_prompts/p512.jsonl 8
g "pp128 int8   " "OV_GPU_TERNOCL_INT2_INT8_PREFILL=1" $W/genai_prompts/p128.jsonl 8
g "pp512 int8   " "OV_GPU_TERNOCL_INT2_INT8_PREFILL=1" $W/genai_prompts/p512.jsonl 8
g "tg32         " ""                                   $W/genai_prompts/p128.jsonl 32
g "tg128        " ""                                   $W/genai_prompts/p128.jsonl 128
echo "## DONE"
