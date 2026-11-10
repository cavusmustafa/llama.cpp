#!/usr/bin/env python3
"""egsum.py <exec_graph.xml>: count exec-graph layers by (layerType, primitiveType)."""
import sys, re, collections
s = open(sys.argv[1]).read(); c = collections.Counter()
for m in re.finditer(r'<layer [^>]*type="([^"]*)"[^>]*>\s*<data ([^>]*)/>', s):
    d = dict(re.findall(r'(\w+)="([^"]*)"', m.group(2)))
    c[(d.get("layerType", m.group(1)), d.get("primitiveType", ""))] += 1
for k, v in sorted(c.items(), key=lambda x: -x[1]): print(f"{v:6d} {k[0]:28s} {k[1]}")
