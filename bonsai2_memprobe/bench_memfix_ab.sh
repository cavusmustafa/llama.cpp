#!/bin/bash
# bench_memfix_ab.sh: same-session A/B of ggml-openvino speed, OpenVINO master (A, install-master)
# vs OpenVINO master + the 3 compile-memory fixes (B, build-master rebuilt with
# memprobe/ov_compile_memory_fixes.patch, same Release flags). Rounds alternate A,B,A,B so drift
# shows up as A-vs-A disagreement. Decode measured with -n alone (pp-before-tg contaminates tg).
set +u
ROOT=/home/mcavus/llama.cpp/gemma4
M=$ROOT/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf
BIN=$ROOT/bonsai-ov/build-master/bin
OV=$ROOT/openvino
source $OV/install-master/setupvars.sh >/dev/null 2>&1
BASE_LD=$LD_LIBRARY_PATH
PATCH_LD=$OV/build-master/bin/intel64/Release:$BASE_LD
ROUNDS=${ROUNDS:-2}
f(){ sed -n 's/.*| *\([pt][pg][0-9]*\) *| *\([0-9.]*\) ± \([0-9.]*\).*/\1 \2 \3/p'; }
run(){ # side, label, env...
  local side=$1 lbl=$2; shift 2
  local ld=$BASE_LD; [ $side = B ] && ld=$PATCH_LD
  env LD_LIBRARY_PATH=$ld GGML_OPENVINO_DEVICE=GPU "$@" $BIN/llama-bench -m $M -p 128,512 -n 0 -r 3 -ngl 99 2>/dev/null | f | sed "s/^/RES $side $lbl /"
  for n in 32 128; do
    env LD_LIBRARY_PATH=$ld GGML_OPENVINO_DEVICE=GPU "$@" $BIN/llama-bench -m $M -p 0 -n $n -r 5 -ngl 99 2>/dev/null | f | sed "s/^/RES $side $lbl /"
  done
}
TUNED="GGML_OPENVINO_ENABLE_KV_4D=1 GGML_OPENVINO_KV_SLICE=0"
MEM="$TUNED GGML_OPENVINO_MEMORY_OPTIMIZE=1 GGML_OPENVINO_SPILL_DIR=$ROOT/memprobe/spill"
for r in $(seq 1 $ROUNDS); do
  for side in A B; do
    echo "# round $r side $side $(date +%T)"
    run $side tuned  $TUNED
    run $side mem16  $MEM
    env LD_LIBRARY_PATH=$([ $side = B ] && echo $PATCH_LD || echo $BASE_LD) GGML_OPENVINO_DEVICE=GPU $TUNED OV_GPU_TERNOCL_INT2_INT8_PREFILL=1 \
      $BIN/llama-bench -m $M -p 128,512 -n 0 -r 3 -ngl 99 2>/dev/null | f | sed "s/^/RES $side int8 /"
  done
done
echo "# DONE $(date +%T)"
