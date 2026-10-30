#!/bin/bash
# Bonsai 2 27B through real OpenVINO GenAI (VLMPipeline via llm_bench) on the
# latest PR #38460 head. Prefill t/s = input tokens / 1st-token latency.
set +u
W=/home/mcavus/llama.cpp/gemma4/ov-bonsai2
source /home/mcavus/llama.cpp/gemma4/openvino/install-pr38460-latest/setupvars.sh >/dev/null 2>&1
cd $W/openvino.genai/tools/llm_bench
run() { # label, extra-env, prompt-file, new-tokens
  env $2 $W/bench-venv/bin/python benchmark.py -m $W/bonsai2-27b-genai -d GPU \
    -pf $3 --disable_prompt_permutation -ic $4 -n 3 2>&1 | grep "\[Average\]" | sed "s/^.*P\[0\]/$1:/"
}
run "pp128 default" ""                                   $W/genai_prompts/p128.jsonl 8
run "pp512 default" ""                                   $W/genai_prompts/p512.jsonl 8
run "pp128 int8   " "OV_GPU_TERNOCL_INT2_INT8_PREFILL=1" $W/genai_prompts/p128.jsonl 8
run "pp512 int8   " "OV_GPU_TERNOCL_INT2_INT8_PREFILL=1" $W/genai_prompts/p512.jsonl 8
run "tg32         " ""                                   $W/genai_prompts/p128.jsonl 32
run "tg128        " ""                                   $W/genai_prompts/p128.jsonl 128
