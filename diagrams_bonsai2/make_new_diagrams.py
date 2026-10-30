#!/usr/bin/env python3
"""Regenerate the 2026-10-17 Bonsai 2 diagrams (11, 12, 13).

House style of diagrams_bonsai2/: 1000px wide, white surface, Segoe UI for prose and
ui-monospace for identifiers. Categorical hues are blue/orange/violet/green in that
fixed order - validated with the dataviz validator (all checks pass, worst adjacent
pair dE 19.2 deutan / 25.9 normal). Grey is a reference/baseline annotation, never a
category, and every reference bar is direct-labelled.
"""

FONT = "Segoe UI, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, Consolas, monospace"

INK, MUTED, RULE = "#1b1f24", "#6b7480", "#8a93a0"
SURF, PANEL = "#ffffff", "#f7f9fb"
BLUE, ORANGE, VIOLET, GREEN = "#2f6fb3", "#c77c1e", "#6b4fa8", "#2d8a4e"
RED = "#c0392b"
GHOST = "#c7cdd6"   # "before" reference bars - annotation, not a series


def head(w, h, title, sub):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">',
        f'<rect width="{w}" height="{h}" fill="{SURF}"/>',
        f'<text x="24" y="34" font-size="20" font-weight="600" fill="{INK}">{title}</text>',
        f'<text x="24" y="56" font-size="13" fill="{MUTED}">{sub}</text>',
    ]


def legend(items, y, x0=24, gap=250):
    out = []
    for i, (color, label) in enumerate(items):
        x = x0 + i * gap
        out.append(f'<rect x="{x}" y="{y}" width="14" height="14" rx="2" fill="{color}"/>')
        out.append(f'<text x="{x + 20}" y="{y + 12}" font-size="11.5" fill="{INK}">{label}</text>')
    return out


def bar_group(y, label, rows, xmax, x_axis=215, width=720, bar_h=19, gap=5, unit=""):
    """rows: list of (value, colour, series_label). Direct-labels every bar."""
    out = [f'<text x="24" y="{y}" font-size="14" font-weight="600" fill="{INK}">{label}</text>']
    top = y + 12
    out.append(f'<line x1="{x_axis}" y1="{top}" x2="{x_axis}" y2="{top + len(rows) * (bar_h + gap)}" stroke="{RULE}"/>')
    for i, (val, colour, slab) in enumerate(rows):
        by = top + i * (bar_h + gap)
        w = max(2.0, val / xmax * width)
        out.append(f'<text x="{x_axis - 10}" y="{by + bar_h - 5}" font-size="11.5" text-anchor="end" fill="{INK}">{slab}</text>')
        # 4px rounded data-end, anchored to the baseline
        out.append(f'<rect x="{x_axis + 1}" y="{by}" width="{w:.1f}" height="{bar_h}" rx="4" fill="{colour}"/>')
        out.append(f'<rect x="{x_axis + 1}" y="{by}" width="{min(w, 6):.1f}" height="{bar_h}" fill="{colour}"/>')
        out.append(f'<text x="{x_axis + w + 9:.1f}" y="{by + bar_h - 5}" font-size="11.5" font-weight="600" fill="{INK}">{val:g}{unit}</text>')
    return out, top + len(rows) * (bar_h + gap)


# ---------------------------------------------------------------- 11: results
def results():
    w, h = 1000, 700
    s = head(w, h, "Measured results - Bonsai 2 27B PQ2_0, latest PR #38460 head",
             "Intel Arc B390 iGPU (Xe3), one OpenVINO build (4185ce2f15) for every bar. Decode measured with -n alone. Higher is better.")
    s += legend([(VIOLET, "OpenVINO GenAI (VLMPipeline)"), (BLUE, "ggml-openvino (this work)"),
                 (ORANGE, "ggml-vulkan"), (GREEN, "int8 prefill variant")], 72, gap=232)

    g, yy = bar_group(124, "Prefill, 512-token prompt  (t/s)", [
        (715.1, GREEN,  "GenAI int8"),
        (657.9, VIOLET, "GenAI"),
        (575.1, GREEN,  "ggml-ov int8"),
        (544.2, BLUE,   "ggml-openvino"),
        (169.5, ORANGE, "vulkan"),
    ], xmax=740)
    s += g

    g, yy = bar_group(yy + 34, "Prefill, 128-token prompt  (t/s)", [
        (638.2, GREEN,  "GenAI int8"),
        (545.8, GREEN,  "ggml-ov int8"),
        (256.2, VIOLET, "GenAI"),
        (236.5, BLUE,   "ggml-openvino"),
        (171.2, ORANGE, "vulkan"),
    ], xmax=740)
    s += g

    g, yy = bar_group(yy + 34, "Decode, 128 tokens  (t/s)", [
        (14.53, VIOLET, "GenAI"),
        (13.21, BLUE,   "ggml-openvino"),
        (12.70, ORANGE, "vulkan"),
    ], xmax=16)
    s += g

    yy += 26
    s.append(f'<rect x="24" y="{yy}" width="952" height="96" rx="6" fill="{PANEL}"/>')
    s.append(f'<text x="40" y="{yy + 25}" font-size="13" font-weight="600" fill="{INK}">Reading it</text>')
    for i, line in enumerate([
        "GenAI leads ggml-openvino by 1.10x on decode and 1.21x on pp512; ggml-openvino leads Vulkan by 3.21x on pp512.",
        "int8 prefill = OV_GPU_TERNOCL_INT2_INT8_PREFILL=1 (prefill only, 2.3x at pp128). The old OV_TERNOCL_... name is ignored here.",
    ]):
        s.append(f'<text x="40" y="{yy + 48 + i * 20}" font-size="12.5" fill="{INK}">{line}</text>')
    s.append(f'<text x="40" y="{yy + 88}" font-size="11.5" fill="{MUTED}">'
             f'GenAI prefill = prompt tokens / first-token latency (llm_bench); decode = 2nd-token throughput.</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------- 12: GDN state round-trip fix
def gdn_state():
    w, h = 1000, 620
    s = head(w, h, "The fix: keep the GDN recurrent state in the op's own layout",
             "48 GatedDeltaNet layers x 2 permutes of a 3.1 MB state, every token - 1.73 ms of a 76.7 ms decode step.")

    def panel(x, y, title, colour, rows, note, note_fill):
        o = [f'<rect x="{x}" y="{y}" width="452" height="300" rx="8" fill="{PANEL}" stroke="#e2e7ee"/>',
             f'<text x="{x + 18}" y="{y + 28}" font-size="14" font-weight="600" fill="{colour}">{title}</text>']
        by = y + 50
        for i, (txt, kind) in enumerate(rows):
            fill = {"op": "#ffffff", "cost": "#fdecea", "free": "#eaf4ee"}[kind]
            stroke = {"op": "#d7dde6", "cost": RED, "free": GREEN}[kind]
            o.append(f'<rect x="{x + 18}" y="{by}" width="416" height="34" rx="5" fill="{fill}" stroke="{stroke}"/>')
            o.append(f'<text x="{x + 30}" y="{by + 22}" font-size="12" font-family="{MONO}" fill="{INK}">{txt}</text>')
            if i < len(rows) - 1:
                o.append(f'<path d="M{x + 226} {by + 34} L{x + 226} {by + 44}" stroke="{RULE}" stroke-width="1.5" marker-end="url(#a12)"/>')
            by += 44
        o.append(f'<text x="{x + 18}" y="{y + 288}" font-size="12" font-weight="600" fill="{note_fill}">{note}</text>')
        return o

    s.append(f'<defs><marker id="a12" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
             f'<path d="M0 0 L10 5 L0 10 z" fill="{RULE}"/></marker></defs>')

    s += panel(24, 84, "Before - ggml layout in the buffer", RED, [
        ("ggml cache_s_l*   [B,H,value,key]", "op"),
        ("Transpose {0,1,3,2}", "cost"),
        ("GatedDeltaNet   [B,H,key,value]", "op"),
        ("Transpose {0,1,3,2}   (back again)", "cost"),
        ("ggml cache_s_l*   [B,H,value,key]", "op"),
    ], "2 permutes x 48 layers = 96 ops, 1.73 ms/token", RED)

    s += panel(524, 84, "After - the op's layout in the buffer", GREEN, [
        ("ggml cache_s_l*   [B,H,key,value]", "op"),
        ("(no transpose)", "free"),
        ("GatedDeltaNet   [B,H,key,value]", "op"),
        ("(no transpose)", "free"),
        ("ggml cache_s_l*   [B,H,key,value]", "op"),
    ], "0 ops, 0 ms - decode 13.05 -> 13.45 t/s", GREEN)

    y = 410
    s.append(f'<rect x="24" y="{y}" width="952" height="176" rx="8" fill="{PANEL}"/>')
    s.append(f'<text x="40" y="{y + 26}" font-size="13" font-weight="600" fill="{INK}">Why this is safe</text>')
    for i, line in enumerate([
        "The state is read from and written straight back to the same ggml buffer, and nothing else observes it in between,",
        "so the buffer can simply hold the op's layout instead of ggml's. Guarded on the two state dims being equal - otherwise",
        "skipping the permute would also change the shape ggml allocated. Bonsai 2's state is square (128x128), so it applies.",
        "Every graph touching the buffer must agree: instrumented 144/144 GDN translations reported the native-layout path.",
    ]):
        s.append(f'<text x="40" y="{y + 52 + i * 21}" font-size="12.5" fill="{INK}">{line}</text>')
    s.append(f'<text x="40" y="{y + 150}" font-size="12" fill="{MUTED}">'
             f'OpenVINO\'s own GGUF frontend made the same change upstream in 630516e048, behind a split_outputs flag.</text>')
    s.append("</svg>")
    return "\n".join(s)


# ------------------------------------------------- 13: where decode time goes
def decode_budget():
    w, h = 1000, 640
    s = head(w, h, "Where a decode step actually goes",
             "Per-token op profile (GGML_OPENVINO_PROFILING=2), 2622 executed ops. The GPU is ~100% busy throughout.")

    total = 76.19
    items = [
        ("FullyConnectedCompressed - the ternary GEMVs", 65.73, BLUE, "257"),
        ("Convert - f32 state boundary", 2.49, ORANGE, "275"),
        ("GatedDeltaNet", 1.90, VIOLET, "48"),
        ("Transpose", 1.08, GREEN, "177"),
        ("FullyConnected - BF16 ssm_alpha/beta", 1.06, RULE, "97"),
        ("Result - cache_r/cache_s writeback", 1.06, RULE, "96"),
        ("RMS", 0.69, RULE, "209"),
        ("StridedSlice / Concat / Add / rest", 2.18, RULE, "1463"),
    ]
    x0, wid, xmax = 420, 500, 70.0
    y = 96
    s.append(f'<text x="24" y="{y}" font-size="14" font-weight="600" fill="{INK}">Per-token cost by op  (ms)</text>')
    s.append(f'<text x="{x0 + wid - 40}" y="{y}" font-size="11.5" fill="{MUTED}">n ops</text>')
    y += 14
    for label, ms, colour, n in items:
        bw = max(2.0, ms / xmax * wid)
        s.append(f'<text x="{x0 - 12}" y="{y + 15}" font-size="11.5" text-anchor="end" fill="{INK}">{label}</text>')
        s.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="20" rx="4" fill="{colour}"/>')
        s.append(f'<rect x="{x0}" y="{y}" width="{min(bw, 6):.1f}" height="20" fill="{colour}"/>')
        s.append(f'<text x="{x0 + bw + 9:.1f}" y="{y + 15}" font-size="11.5" font-weight="600" fill="{INK}">{ms:g}</text>')
        s.append(f'<text x="{x0 + wid + 52}" y="{y + 15}" font-size="11" text-anchor="end" font-family="{MONO}" fill="{MUTED}">{n}</text>')
        y += 26

    y += 16
    s.append(f'<rect x="24" y="{y}" width="952" height="104" rx="8" fill="{PANEL}"/>')
    s.append(f'<text x="40" y="{y + 26}" font-size="13" font-weight="600" fill="{INK}">The GEMVs are not the problem</text>')
    for i, line in enumerate([
        "6.86 GB of weights in 65.7 ms is 106 GB/s - 69% of this machine's 153.6 GB/s peak (8 x 16-bit LPDDR5X-9600), and both",
        "backends run the same TernOCL kernels over the same u2 weights. The whole remaining gap is the ~10 ms of non-GEMV",
        "graph scaffolding, where the mean op costs 4.9 us and so op *count* is what matters, not op efficiency.",
    ]):
        s.append(f'<text x="40" y="{y + 50 + i * 20}" font-size="12.5" fill="{INK}">{line}</text>')

    y += 124
    s.append(f'<text x="24" y="{y}" font-size="14" font-weight="600" fill="{INK}">Decode step, side by side  (ms/token)</text>')
    y += 14
    sx, sw, smax = 200, 380, 80.0
    for label, gemv, over, tone in [("native OpenVINO", 64.7, 2.5, VIOLET), ("ggml-openvino", 65.7, 10.5, BLUE)]:
        gw, ow = gemv / smax * sw, over / smax * sw
        s.append(f'<text x="{sx - 12}" y="{y + 15}" font-size="11.5" text-anchor="end" fill="{INK}">{label}</text>')
        s.append(f'<rect x="{sx}" y="{y}" width="{gw:.1f}" height="20" rx="4" fill="{tone}"/>')
        s.append(f'<rect x="{sx}" y="{y}" width="6" height="20" fill="{tone}"/>')
        # 2px surface gap between adjacent fills
        s.append(f'<rect x="{sx + gw + 2:.1f}" y="{y}" width="{ow:.1f}" height="20" rx="4" fill="{ORANGE}"/>')
        s.append(f'<text x="{sx + gw + ow + 14:.1f}" y="{y + 15}" font-size="11.5" font-weight="600" fill="{INK}">'
                 f'{gemv:g} ms GEMV  +  {over:g} ms overhead</text>')
        y += 28
    s.append(f'<rect x="{sx}" y="{y + 4}" width="12" height="12" rx="2" fill="{ORANGE}"/>')
    s.append(f'<text x="{sx + 18}" y="{y + 14}" font-size="11.5" fill="{MUTED}">'
             f'non-GEMV overhead - same GEMV term on both sides, so this band is the entire 1.12x.</text>')
    s.append("</svg>")
    return "\n".join(s)


import pathlib
here = pathlib.Path(__file__).parent
for name, svg in [("11-results-2026-10-17.svg", results()),
                  ("12-gdn-state-roundtrip.svg", gdn_state()),
                  ("13-decode-budget.svg", decode_budget())]:
    (here / name).write_text(svg)
    print("wrote", name, len(svg), "bytes")
