#!/bin/bash
# breakdown.sh <label> <env...>: unrestricted run inside cgroup memprobe, sampling every 0.5 s:
# system-wide taken (baseline MemAvailable - now), cgroup anon, cgroup file. Reports values at the
# moment of system peak, and in steady decode.
CG=/sys/fs/cgroup/memprobe; label=$1; shift
R=/home/mcavus/llama.cpp/gemma4; M=$R/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
echo max > $CG/memory.max
base=$(awk '/MemAvailable/{print $2}' /proc/meminfo); : > /tmp/bd.csv
bash -c "sudo -n sh -c \"echo \$\$ > $CG/cgroup.procs\" || exit 99; set +u; source $R/openvino/install-master/setupvars.sh >/dev/null 2>&1;
  exec env GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_ENABLE_KV_4D=1 GGML_OPENVINO_KV_SLICE=0 $* \
  $R/bonsai-ov/build-master/bin/llama-cli -m $M -ngl 99 -c 4096 -p 'List the first 10 prime numbers in order.' -n 400 --temp 0 --single-turn $LOADMODE" > /tmp/bd.out 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  an=$(awk '$1=="anon"{print $2}' $CG/memory.stat); fi=$(awk '$1=="file"{print $2}' $CG/memory.stat)
  echo "$(( (base-a)*1024 )) $an $fi" >> /tmp/bd.csv; sleep 0.5
done; wait $pid
python3 - "$label" <<'PY'
import sys, statistics
rows = [tuple(int(x) for x in l.split()) for l in open("/tmp/bd.csv") if len(l.split()) == 3]
G = 1 << 30
i = max(range(len(rows)), key=lambda k: rows[k][0])
sysp, an, fi = rows[i]
drv = max(0, sysp - an - fi)
tail = rows[len(rows) * 2 // 3:]
st = [statistics.median(c) for c in zip(*tail)]
print(f"{sys.argv[1]}")
print(f"  at peak : system {sysp/G:5.1f}  = anon {an/G:5.1f} + file(reclaimable) {fi/G:5.1f} + not-in-cgroup(driver) {drv/G:5.1f} GiB"
      f"   -> must be resident ~{(an+drv)/G:4.1f} GiB")
print(f"  steady  : system {st[0]/G:5.1f}  = anon {st[1]/G:5.1f} + file {st[2]/G:5.1f} + driver {max(0,st[0]-st[1]-st[2])/G:5.1f} GiB")
PY
grep -q "^2, 3, 5, 7, 11, 13, 17, 19, 23, 29" /tmp/bd.out && echo "  output correct" || echo "  OUTPUT WRONG"
