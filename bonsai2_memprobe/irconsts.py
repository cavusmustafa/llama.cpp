#!/usr/bin/env python3
"""irconsts.py <xml>: sum Const bytes by element_type, and list top-N biggest consts."""
import sys, re, collections
s = open(sys.argv[1]).read()
tot = collections.Counter(); cnt = collections.Counter(); big = []
for m in re.finditer(r'<layer id="(\d+)" name="([^"]*)" type="Const"[^>]*>\s*<data([^/]*)/>', s):
    d = dict(re.findall(r'(\w+)="([^"]*)"', m.group(3)))
    sz = int(d.get("size", 0)); et = d.get("element_type"); tot[et] += sz; cnt[et] += 1
    big.append((sz, et, d.get("shape"), m.group(2)[:70]))
for et in tot: print(f"{et:6s} n={cnt[et]:6d} {tot[et]/2**30:7.3f} GiB")
print(f"total {sum(tot.values())/2**30:.3f} GiB")
for b in sorted(big, reverse=True)[:12]: print(f"  {b[0]/2**20:9.1f} MiB {b[1]} {b[2]} {b[3]}")
