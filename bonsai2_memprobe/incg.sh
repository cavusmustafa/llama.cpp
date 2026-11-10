#!/bin/bash
# incg.sh <limit|max> <label> <env...> : run the Bonsai llama-cli test inside cgroup "memprobe".
CG=/sys/fs/cgroup/memprobe; lim=$1; label=$2; shift 2
R=/home/mcavus/llama.cpp/gemma4; M=$R/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
echo $lim > $CG/memory.max; echo 0 > $CG/memory.swap.max
echo 0 > $CG/memory.peak 2>/dev/null   # reset peak (kernel >= 6.12)
oom0=$(awk '/^oom_kill /{print $2}' $CG/memory.events)
bash -c "sudo -n sh -c \"echo \$\$ > $CG/cgroup.procs\" || exit 99; set +u; source $R/openvino/install-master/setupvars.sh >/dev/null 2>&1;
  exec env GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_ENABLE_KV_4D=1 GGML_OPENVINO_KV_SLICE=0 $* \
  $R/bonsai-ov/build-master/bin/llama-cli -m $M -ngl 99 -c 4096 -p 'List the first 10 prime numbers in order.' -n 120 --temp 0 --single-turn" > /tmp/cg.out 2>&1
rc=$?
oom=$(( $(awk '/^oom_kill /{print $2}' $CG/memory.events) - oom0 ))
ok=$(grep -c "^2, 3, 5, 7, 11, 13, 17, 19, 23, 29" /tmp/cg.out)
gen=$(grep -oE "Generation: [0-9.]+ t/s" /tmp/cg.out)
printf "%-38s limit=%-5s rc=%-3d oom_kills=%d correct=%s cgroup-peak=%5.1f GiB  %s\n" "$label" "$lim" $rc $oom \
  $([ "$ok" -gt 0 ] && echo yes || echo NO) $(echo "$(cat $CG/memory.peak)/1073741824" | bc -l) "$gen"
