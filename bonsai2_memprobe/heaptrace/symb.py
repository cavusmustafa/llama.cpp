#!/usr/bin/env python3
"""symb.py <heaptrace.txt> [nstacks]: addr2line the module+offset frames of the top stacks."""
import sys, subprocess, re, functools
@functools.lru_cache(maxsize=None)
def a2l(mod, off):
    try:
        out = subprocess.run(["addr2line", "-f", "-C", "-i", "-e", mod, off], capture_output=True, text=True, timeout=60).stdout.split("\n")
        fn = out[0][:110]; loc = out[1].split("/")[-1] if len(out) > 1 else ""
        return f"{fn}  [{loc}]"
    except Exception as e: return "?"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6; k = 0
for line in open(sys.argv[1]):
    if line.startswith("STACK"):
        k += 1
        if k > n: break
        print(line.strip())
    elif line.startswith("  ") and k <= n:
        mod, off = line.split()
        m = mod.split("/")[-1]
        if any(x in m for x in ("libc.so", "libstdc++", "heaptrace")): continue
        print(f"    {m:32s} {a2l(mod, off)}")
