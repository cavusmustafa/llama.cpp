#!/usr/bin/env python3
"""summarize_ab.py <log>: per (config, test) mean of A and B over rounds, B/A ratio, and the
A-vs-A / B-vs-B spread across rounds (drift/noise floor to compare the effect against)."""
import sys, collections, statistics as st
d = collections.defaultdict(list)
for l in open(sys.argv[1]):
    p = l.split()
    if p[:1] == ["RES"]:
        d[(p[2], p[3], p[1])].append(float(p[4]))
keys = sorted({(c, t) for c, t, _ in d}, key=lambda k: (k[0], k[1][:2] == "tg", int(k[1][2:])))
print(f"{'config':7s} {'test':6s} {'A (base)':>18s} {'B (fixes)':>18s} {'B/A':>7s} {'A spread':>9s} {'B spread':>9s}")
for c, t in keys:
    a, b = d[(c, t, "A")], d[(c, t, "B")]
    if not a or not b:
        continue
    sp = lambda v: (max(v) - min(v)) / st.mean(v) * 100 if len(v) > 1 else 0
    print(f"{c:7s} {t:6s} {st.mean(a):9.2f} {str([round(x, 2) for x in a]):>8s} {st.mean(b):9.2f} {str([round(x, 2) for x in b]):>8s}"
          f" {st.mean(b) / st.mean(a):7.3f} {sp(a):8.1f}% {sp(b):8.1f}%")
