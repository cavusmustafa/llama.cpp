#!/usr/bin/env python3
"""phases.py <label> <logfile> -- cmd...: run cmd, sample every 100 ms (system taken = baseline
MemAvailable - now; process anon and file RSS), then attribute the memory to the OVPHASE markers
the instrumented backend prints on stderr."""
import os, subprocess, sys, time
label, log = sys.argv[1], sys.argv[2]
cmd = sys.argv[sys.argv.index("--") + 1:]
def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1]) * 1024
def gtt(pid):
    # exact per-process GPU buffer memory from xe DRM fdinfo (sum over /dev/dri fds; dedup by drm-client-id)
    tot = {}
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                if "/dev/dri" not in os.readlink(f"/proc/{pid}/fd/{fd}"): continue
                cid, v = None, 0
                for l in open(f"/proc/{pid}/fdinfo/{fd}"):
                    if l.startswith("drm-client-id"): cid = l.split()[1]
                    elif l.startswith("drm-total-gtt") or l.startswith("drm-total-system"): v += int(l.split()[1]) * 1024
                tot[cid] = v
            except OSError: pass
    except OSError: pass
    return sum(tot.values())
def rss(pid):
    d = {}
    try:
        for l in open(f"/proc/{pid}/status"):
            if l.startswith(("RssAnon", "RssFile", "RssShmem")): d[l.split(":")[0]] = int(l.split()[1]) * 1024
    except FileNotFoundError: pass
    return d.get("RssAnon", 0), d.get("RssFile", 0) + d.get("RssShmem", 0)
subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
base = avail(); samples = []
with open(log, "w") as f:
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    while p.poll() is None:
        a, fi = rss(p.pid)
        samples.append((time.time() * 1000, base - avail(), a, fi, gtt(p.pid))); time.sleep(0.1)
marks = []
import re
for m in re.finditer(r"OVPHASE (\d+) ([^\n\r]*)", open(log, errors="replace").read()):
    marks.append((int(m.group(1)), m.group(2).strip()))
G = 1 << 30
t0 = samples[0][0]
def at(ts):  # sample nearest to ts
    return min(samples, key=lambda s: abs(s[0] - ts))
print(f"== {label}  (run {(samples[-1][0]-t0)/1000:.0f}s)")
print(f"   {'t(s)':>6}  {'system':>7} {'anon':>6} {'file':>6} {'drv~':>6} {'gtt':>6}   event")
pk = max(samples, key=lambda s: s[1]); pg = max(samples, key=lambda s: s[4]); pa = max(samples, key=lambda s: s[2])
pga = max(samples, key=lambda s: s[2] + s[4])
evs = [(ts, n) for ts, n in marks] + [(pk[0], "<<< SYSTEM PEAK"), (pg[0], "<<< GTT PEAK"), (pga[0], "<<< ANON+GTT PEAK (process-exact)")]
for ts, n in sorted(evs):
    s = at(ts)
    print(f"   {(ts-t0)/1000:6.1f}  {s[1]/G:7.1f} {s[2]/G:6.1f} {s[3]/G:6.1f} {max(0,s[1]-s[2])/G:6.1f} {s[4]/G:6.1f}   {n}")
if os.environ.get("SERIES"):
    step = float(os.environ["SERIES"])
    nxt = 0
    for s_ in samples:
        t = (s_[0] - t0) / 1000
        if t >= nxt:
            print(f"   ts {t:6.1f}  sys {s_[1]/G:5.1f} anon {s_[2]/G:5.1f} gtt {s_[4]/G:5.1f}"); nxt = t + step
# peak inside each phase window
names = [n for _, n in marks]
def win(a, b):
    ta = next((t for t, n in marks if n == a), None); tb = next((t for t, n in marks if n == b), None)
    if ta is None or tb is None: return None
    w = [s for s in samples if ta <= s[0] <= tb]
    return max(w, key=lambda s: s[1]) if w else at(ta)
for a, b, what in [("set_tensor_weight #1", "set_tensor_weight #851", "weight load (set_tensor)"),
                   ("weight_nodes_begin", "weight_nodes_end", "create_weight_nodes"),
                   ("convert_begin", "convert_end", "FrontEnd::convert"),
                   ("compile_begin", "compile_end", "compile_model")]:
    w = win(a, b)
    if w: print(f"   max during {what:26s}: system {w[1]/G:5.1f}  anon {w[2]/G:5.1f}  driver~ {max(0,w[1]-w[2])/G:5.1f} GiB")
ce = next((t for t, n in marks if n == "compile_end"), None)
if ce:
    w = [s for s in samples if s[0] > ce]
    m = max(w, key=lambda s: s[1]); print(f"   max after compile (first inference + decode): system {m[1]/G:5.1f} GiB")
