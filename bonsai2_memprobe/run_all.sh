#!/bin/bash
set +u
R=/home/mcavus/llama.cpp/gemma4; M=$R/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; P=$R/memprobe/memprobe.sh
( source $R/openvino/install-master/setupvars.sh >/dev/null 2>&1
  export GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_ENABLE_KV_4D=1 GGML_OPENVINO_KV_SLICE=0
  $P "ggml-openvino llama-bench pp512+tg128"   $R/bonsai-ov/build-master/bin/llama-bench -m $M -p 512 -n 128 -r 2 -ngl 99
  $P "ggml-openvino llama-bench int8 pp512"    env OV_GPU_TERNOCL_INT2_INT8_PREFILL=1 $R/bonsai-ov/build-master/bin/llama-bench -m $M -p 512 -n 0 -r 2 -ngl 99
  $P "ggml-openvino llama-cli ctx 4096"        $R/bonsai-ov/build-master/bin/llama-cli -m $M -ngl 99 -c 4096 -p "Explain photosynthesis." -n 128 --temp 0 --single-turn
  $P "ggml-openvino llama-cli ctx 32768"       $R/bonsai-ov/build-master/bin/llama-cli -m $M -ngl 99 -c 32768 -p "Explain photosynthesis." -n 128 --temp 0 --single-turn
  W=$R/ov-bonsai2; cd $W/openvino.genai/tools/llm_bench
  $P "GenAI llm_bench pp512 + 128 tok"         $W/bench-venv/bin/python benchmark.py -m $W/bonsai2-27b-genai -d GPU -pf $W/genai_prompts/p512.jsonl --disable_prompt_permutation -ic 128 -n 1 )
( source $R/1.4.350.1/setup-env.sh >/dev/null 2>&1
  $P "ggml-vulkan llama-bench pp512+tg128"     $R/bonsai-ov/build-vk/bin/llama-bench -m $M -p 512 -n 128 -r 2 -ngl 99 )
echo DONE
