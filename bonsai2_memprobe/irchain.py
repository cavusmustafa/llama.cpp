#!/usr/bin/env python3
"""irchain.py <xml> <n>: for the first n MatMuls, print the upstream subgraph of input 1 (weights)."""
import sys, re
s = open(sys.argv[1]).read(); n = int(sys.argv[2])
L = {}
for m in re.finditer(r'<layer id="(\d+)" name="([^"]*)" type="([^"]*)"[^>]*>(.*?)</layer>', s, re.S):
    d = re.search(r'<data([^>]*)/?>', m.group(4)); attrs = dict(re.findall(r'(\w+)="([^"]*)"', d.group(1))) if d else {}
    outs = re.findall(r'precision="([^"]*)"[^>]*>(.*?)</port>', m.group(4).split("<output>")[1], re.S) if "<output>" in m.group(4) else []
    oshape = [",".join(re.findall(r"<dim>([^<]*)</dim>", o[1])) for o in outs]
    L[m.group(1)] = (m.group(3), attrs, [o[0] for o in outs], oshape, m.group(2))
E = {}
for m in re.finditer(r'<edge from-layer="(\d+)" from-port="(\d+)" to-layer="(\d+)" to-port="(\d+)"', s):
    E.setdefault(m.group(3), []).append((int(m.group(4)), m.group(1)))
def show(i, depth, seen):
    t, a, p, sh, nm = L[i]
    extra = a.get("element_type", "") + " " + a.get("shape", "") if t == "Const" else ""
    print("  " * depth + f"{t} {p} [{';'.join(sh)}] {extra} {nm[:40]}")
    if t == "Const" or depth > 8 or i in seen: return
    seen.add(i)
    for port, src in sorted(E.get(i, [])):
        if L[src][0] in ("Parameter",): print("  " * (depth + 1) + "Parameter"); continue
        show(src, depth + 1, seen)
k = 0
for i, (t, a, p, sh, nm) in L.items():
    if t == "MatMul":
        print(f"## MatMul {nm} attrs={a}"); srcs = dict(E[i]); show(srcs[1], 1, set()); k += 1
        if k >= n: break
