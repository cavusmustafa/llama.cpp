#!/bin/bash
set +u
R=/home/mcavus/llama.cpp/gemma4; M=$R/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; T=$R/memprobe/timeline.sh
source $R/openvino/install-master/setupvars.sh >/dev/null 2>&1
CLI="$R/bonsai-ov/build-master/bin/llama-cli -m $M -ngl 99 -c 4096 -p 'List the first 10 prime numbers in order.' -n 400 --temp 0 --single-turn"
base="GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_ENABLE_KV_4D=1 GGML_OPENVINO_KV_SLICE=0"
run(){ $T "$1" bash -c "env $base $2 $CLI"; }
run "default"                       ""
run "SPILL_DIR"                     "GGML_OPENVINO_SPILL_DIR=$R/memprobe/spill"
run "MEMORY_OPTIMIZE=1"             "GGML_OPENVINO_MEMORY_OPTIMIZE=1"
run "MEMORY_OPTIMIZE=1 + SPILL_DIR" "GGML_OPENVINO_MEMORY_OPTIMIZE=1 GGML_OPENVINO_SPILL_DIR=$R/memprobe/spill"
run "REDUCE_COMPILE_MEM=1 only"     "GGML_OPENVINO_REDUCE_COMPILE_MEM=1"
echo DONE
