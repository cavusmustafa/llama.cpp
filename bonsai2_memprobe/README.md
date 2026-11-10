# bonsai2_memprobe - memory and speed measurement tools for Bonsai 2 27B on ggml-openvino

Used to root-cause the GPU-plugin compile-memory peak (28.2 -> 10.0 GiB). The OpenVINO fixes are
on `cavusmustafa/openvino` branch `gpu-compile-memory-fixes`; the story is in
`../bonsai2_report/BONSAI2_TERNARY_FULL_REPORT.md` (+ standalone `.html`).
Scripts hard-code paths under `/home/mcavus/llama.cpp/gemma4`; adjust `ROOT`/`R` to reuse them.

| file | what |
|---|---|
| `phases.py <label> <log> -- cmd...` | 100 ms timeline of the process's CPU heap (RssAnon) + GPU buffers (xe fdinfo `drm-total-gtt`); prints `ANON+GTT PEAK` (process-exact), attributes `OVPHASE <ms> <name>` markers. `SERIES=<s>` prints the time series |
| `ovcompile.cpp` | pure-OpenVINO `compile_model` reproducer for ggml-openvino IR dumps (`GGML_OPENVINO_DUMP_IR=1`); registers `GatedDeltaNet`. Env: `EXEC_GRAPH`, `KMEM`, `CUT_EMBD_GATHER`, `COMPILE_THREADS` |
| `heaptrace/` | LD_PRELOAD tracker: live bytes per call stack for allocations >= 64 KiB; `symb.py` addr2lines the top stacks (use a RelWithDebInfo plugin) |
| `gathertest.cpp` | u2 GatherCompressed correctness test vs a host reference |
| `irconsts.py`, `irchain.py`, `egsum.py` | IR constant sizes, weight-subgraph chains, exec-graph op/impl counts |
| `hog.py`, `limit16.sh <GiB> <label>` | faithful small-RAM test: mlock `MemTotal - GiB`, run llama-cli under a watchdog, report peak / swap / OOM |
| `bench_memfix_ab.sh`, `summarize_ab.py`, `bench_memfix_ab_20261110.txt` | interleaved A/B speed benchmark (baseline OV vs OV + fixes) and its 2026-11-10 result: speed-neutral |
| `memprobe.sh`, `timeline.sh`, `breakdown.sh`, `incg.sh`, `run_*.sh` | earlier system-wide/cgroup probes. NOTE: cgroups and MemAvailable do not see GPU buffers on an iGPU; prefer `phases.py` |

Compile the C/C++ tools against an OpenVINO install, e.g.:

    g++ -O1 -std=c++17 ovcompile.cpp -o ovcompile -I$OV/runtime/include -I<openvino-src>/src/core/dev_api \
        -L$OV/runtime/lib/intel64 -lopenvino -Wl,-rpath,$OV/runtime/lib/intel64
    gcc -O2 -shared -fPIC heaptrace/heaptrace.c -o heaptrace/libheaptrace.so -ldl -lpthread
