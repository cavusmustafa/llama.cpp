#!/usr/bin/env python3
"""ovcompile.py <xml> [device]: pure-OpenVINO compile_model reproducer, prints OVPHASE markers."""
import sys, time, openvino as ov
def mark(n): print(f"OVPHASE {int(time.time()*1000)} {n}", file=sys.stderr, flush=True)
xml = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "GPU"
core = ov.Core()
mark("read_begin"); m = core.read_model(xml); mark("read_end")
mark("compile_begin"); c = core.compile_model(m, dev); mark("compile_end")
try:
    st = core.get_property(dev, "GPU_MEMORY_STATISTICS")
    print("GPU_MEMORY_STATISTICS", {k: round(v/2**30, 2) for k, v in st.items()}, file=sys.stderr)
except Exception as e: print("memstat err", e, file=sys.stderr)
del m; time.sleep(3); mark("after_del_model")
del c; time.sleep(3); mark("after_del_compiled")
