#!/usr/bin/env python3
"""GPU busy% from the xe PMU via perf_event_open (perf(1) is not built for this kernel).

busy% = engine-active-ticks / engine-total-ticks over the sample window, per engine class.
Usage: gpu_busy.py [seconds]
"""
import ctypes, ctypes.util, os, struct, sys, time, glob

PERF_TYPE = int(open("/sys/bus/event_source/devices/xe_0000_00_02.0/type").read())
EVDIR = "/sys/bus/event_source/devices/xe_0000_00_02.0/events"


def ev_config(name):
    # e.g. "event=0x02" (plus engine_class/instance packed in by us)
    raw = open(os.path.join(EVDIR, name)).read().strip()
    cfg = 0
    for part in raw.split(","):
        if part.startswith("event="):
            cfg |= int(part.split("=")[1], 0)
    return cfg


# from the xe PMU format dir: event:0-11, engine_instance:12-19, engine_class:20-27, gt:60-63
def pack(event_cfg, engine_class, engine_instance):
    return (event_cfg & 0xFFF) | ((engine_instance & 0xFF) << 12) | ((engine_class & 0xFF) << 20)


class PerfEventAttr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32), ("size", ctypes.c_uint32), ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64), ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64), ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32), ("bp_type", ctypes.c_uint32),
        ("bp_addr", ctypes.c_uint64), ("bp_len", ctypes.c_uint64),
        ("branch_sample_type", ctypes.c_uint64), ("sample_regs_user", ctypes.c_uint64),
        ("sample_stack_user", ctypes.c_uint32), ("clockid", ctypes.c_int32),
        ("sample_regs_intr", ctypes.c_uint64), ("aux_watermark", ctypes.c_uint32),
        ("sample_max_stack", ctypes.c_uint16), ("__reserved_2", ctypes.c_uint16),
        ("aux_sample_size", ctypes.c_uint32), ("__reserved_3", ctypes.c_uint32),
    ]


libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def open_counter(config):
    attr = PerfEventAttr()
    attr.type = PERF_TYPE
    attr.size = ctypes.sizeof(PerfEventAttr)
    attr.config = config
    # pid=-1, cpu=0 -> system-wide on cpu0 (uncore-style PMU)
    fd = libc.syscall(298, ctypes.byref(attr), -1, 0, -1, 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), f"perf_event_open failed for config {config:#x}")
    return fd


def read_counter(fd):
    return struct.unpack("Q", os.read(fd, 8))[0]


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    active_cfg, total_cfg = ev_config("engine-active-ticks"), ev_config("engine-total-ticks")
    classes = {0: "render/RCS", 1: "copy/BCS", 4: "compute/CCS"}
    counters = {}
    for cls, label in classes.items():
        for inst in (0, 1):
            try:
                a = open_counter(pack(active_cfg, cls, inst))
                t = open_counter(pack(total_cfg, cls, inst))
                counters[(cls, inst)] = (a, t, label)
            except OSError:
                pass
    if not counters:
        print("no xe engine counters could be opened")
        return
    base = {k: (read_counter(v[0]), read_counter(v[1])) for k, v in counters.items()}
    time.sleep(dur)
    print(f"--- GPU engine busy over {dur:.0f}s ---")
    for k, (afd, tfd, label) in counters.items():
        a1, t1 = read_counter(afd), read_counter(tfd)
        da, dt = a1 - base[k][0], t1 - base[k][1]
        if dt:
            print(f"  {label} inst{k[1]}: {100.0*da/dt:6.2f}% busy   (active {da}, total {dt})")


main()
