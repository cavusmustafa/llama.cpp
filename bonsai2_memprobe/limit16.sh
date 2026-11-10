#!/bin/bash
# limit16.sh <avail_target_GiB> <label>: reserve hugepages so MemAvailable ~= target, run the patched
# llama-cli under a watchdog, report peak / swap / OOM, always release the reservation.
T=$1; L=$2; R=~/llama.cpp/gemma4; M=$R/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; OV=$R/openvino; D=$OV/build-dbgcaps/bin/intel64/RelWithDebInfo
trap 'sudo -n pkill -f "memprobe/hog.py"; sleep 2; echo "released: hog alive=$(pgrep -fc memprobe/hog.py) MemAvailable=$(awk "/MemAvailable/{printf \"%.1f\", \$2/1048576}" /proc/meminfo) GiB"' EXIT
sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
# hog = MemTotal - target: everything else on the box then has to fit in <target> GiB of RAM
tot=$(awk '/MemTotal/{print $2}' /proc/meminfo); H=$(python3 -c "print(round(($tot/1048576) - $T, 2))")
sudo -n python3 $R/memprobe/hog.py $H > /tmp/hog_$L.log 2>&1 & HP=$!
until grep -q -E "READY|failed" /tmp/hog_$L.log 2>/dev/null; do sleep 1; kill -0 $HP 2>/dev/null || break; done
cat /tmp/hog_$L.log; n=0; got=0
echo "[$L] settling 20 s"; sleep 20
echo "[$L] hog ${H} GiB locked (Unevictable $(awk '/Unevictable/{printf "%.1f", $2/1048576}' /proc/meminfo) GiB); MemAvailable now $(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo) GiB"
source $OV/install-master/setupvars.sh >/dev/null
export LD_LIBRARY_PATH=$D:$LD_LIBRARY_PATH GGML_OPENVINO_DEVICE=GPU GGML_OPENVINO_ENABLE_KV_4D=1 GGML_OPENVINO_KV_SLICE=0 GGML_OPENVINO_MEMORY_OPTIMIZE=1 GGML_OPENVINO_SPILL_DIR=$R/memprobe/spill
sw0=$(awk '/pswpin|pswpout/{s=s" "$2} END{print s}' /proc/vmstat); oom0=$(awk '/oom_kill /{print $2}' /proc/vmstat)
python3 - "$L" <<'PY'
import subprocess, sys, time, os
def avail(): return int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 1048576
cmd = [os.path.expanduser("~/llama.cpp/gemma4/bonsai-ov/build-master/bin/llama-cli"), "-m", os.path.expanduser("~/llama.cpp/gemma4/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf"),
       "-ngl", "99", "-c", "4096", "-p", "List the first 10 prime numbers in order.", "-n", "400", "--temp", "0", "--single-turn"]
log = open(f"/tmp/limit16_{sys.argv[1]}.log", "w"); p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
a0 = avail(); lo = a0; killed = False; t0 = time.time(); nxt = 0
def swp(): return {l.split()[0]: int(l.split()[1]) for l in open("/proc/vmstat") if l.startswith(("pswpin", "pswpout"))}
s0 = swp()
while p.poll() is None:
    a = avail(); lo = min(lo, a)
    try:
        st = {l.split(":")[0]: int(l.split()[1]) for l in open(f"/proc/{p.pid}/status") if l.startswith(("VmSwap", "RssAnon"))}
        g = 0
        for fd in os.listdir(f"/proc/{p.pid}/fd"):
            try:
                if "/dev/dri" in os.readlink(f"/proc/{p.pid}/fd/{fd}"):
                    g = max(g, sum(int(l.split()[1]) for l in open(f"/proc/{p.pid}/fdinfo/{fd}") if l.startswith("drm-total-gtt")))
            except OSError: pass
        sw_max = max(globals().get("sw_max", 0), st.get("VmSwap", 0)); pk = max(globals().get("pk", 0), st.get("RssAnon", 0) + g)
        if time.time() - t0 >= nxt:
            s1 = swp(); nxt += 4
            print(f"   t {time.time()-t0:5.0f}s avail {a:5.1f}  ours anon {st.get('RssAnon',0)/1048576:4.1f} gtt {g/1048576:4.1f} swap {st.get('VmSwap',0)/1048576:4.2f}  sys swapout {(s1['pswpout']-s0['pswpout'])*4/1048576:5.2f} in {(s1['pswpin']-s0['pswpin'])*4/1048576:5.2f} GiB", flush=True)
    except (OSError, StopIteration): pass
    if a < 0.6: p.kill(); killed = True; break
    if time.time() - t0 > 900: p.kill(); killed = True; break
    time.sleep(0.1)
p.wait()
print(f"[{sys.argv[1]}] start avail {a0:.1f} GiB, min avail {lo:.2f} GiB, used {a0-lo:.1f} GiB, rc {p.returncode}, watchdog_killed {killed}, {time.time()-t0:.0f}s")
print(f"[{sys.argv[1]}] our process: peak anon+gtt {globals().get('pk',0)/1048576:.1f} GiB, peak VmSwap {globals().get('sw_max',0)/1024:.0f} MiB")
PY
sw1=$(awk '/pswpin|pswpout/{s=s" "$2} END{print s}' /proc/vmstat); oom1=$(awk '/oom_kill /{print $2}' /proc/vmstat)
echo "[$L] pswpin/pswpout before:$sw0 after:$sw1 (pages), oom_kill delta $((oom1-oom0))"
grep -aE "^[0-9]|Prompt:" /tmp/limit16_$L.log | tr '\r' '\n' | grep -av "^\s*$" | tail -2
