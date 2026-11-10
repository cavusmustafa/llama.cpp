#!/usr/bin/env python3
"""hog.py <GiB>: mmap + mlock + touch GiB of anonymous memory (acts as removed RAM), print READY, hold until killed."""
import ctypes, mmap, sys, signal, time
n = int(float(sys.argv[1]) * (1 << 30))
libc = ctypes.CDLL("libc.so.6", use_errno=True)
m = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
addr = ctypes.addressof(ctypes.c_char.from_buffer(m))
step = 1 << 30
for off in range(0, n, step):  # lock in 1 GiB slices so progress is visible and failure is early
    sz = min(step, n - off)
    if libc.mlock(ctypes.c_void_p(addr + off), ctypes.c_size_t(sz)) != 0:
        print(f"mlock failed at {off >> 30} GiB errno {ctypes.get_errno()}", flush=True); sys.exit(1)
print("READY", flush=True)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
while True: time.sleep(1)
