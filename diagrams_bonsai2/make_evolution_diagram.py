#!/usr/bin/env python3
"""Diagram 14: how the Bonsai 2 numbers moved across the four measurement milestones.

Small multiples (decode | prefill), one shared category axis, no dual y-scale. Hues are
the validated house set: violet = native OpenVINO reference, blue = ggml-openvino,
orange = ggml-vulkan. Every bar is direct-labelled; missing cells are marked, not drawn.
"""
import pathlib

FONT = "Segoe UI, Helvetica, Arial, sans-serif"
INK, MUTED, RULE, GRID = "#1b1f24", "#6b7480", "#8a93a0", "#e6e9ee"
PANEL = "#f7f9fb"
BLUE, ORANGE, VIOLET = "#2f6fb3", "#c77c1e", "#6b4fa8"

MILESTONES = [
    ("10-08", "original report", "PR 5c6d4692ee"),
    ("10-17", "decode work", "PR 5c6d4692ee"),
    ("10-30", "PR head", "PR 4185ce2f15"),
    ("10-30", "OV master + sync", "OV 5d07300df3"),
]
# (native reference, ggml-openvino, vulkan); None = not measured at that milestone
DECODE = [(None, 12.24, 12.31), (15.03, 13.46, 12.70), (14.53, 13.21, None), (14.57, 13.28, 12.71)]
PP512 = [(None, 527.3, 165.4), (638.5, 535.1, 169.5), (657.9, 544.2, None), (680.0, 536.1, 168.7)]
SERIES = [(VIOLET, "native OpenVINO reference"), (BLUE, "ggml-openvino"), (ORANGE, "ggml-vulkan")]


def panel(x0, y0, w, h, title, data, ymax, ticks, fmt):
    out = [f'<text x="{x0}" y="{y0 - 14}" font-size="14" font-weight="600" fill="{INK}">{title}</text>']
    base = y0 + h
    for t in ticks:  # recessive grid
        ty = base - t / ymax * h
        out.append(f'<line x1="{x0}" y1="{ty:.1f}" x2="{x0 + w}" y2="{ty:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{x0 - 6}" y="{ty + 4:.1f}" font-size="10.5" text-anchor="end" fill="{MUTED}">{fmt(t)}</text>')
    out.append(f'<line x1="{x0}" y1="{base}" x2="{x0 + w}" y2="{base}" stroke="{RULE}"/>')
    gw = w / len(data)
    bw = 38
    for gi, triple in enumerate(data):
        gx = x0 + gi * gw + (gw - 3 * bw - 4) / 2   # 2px surface gap between bars
        for si, v in enumerate(triple):
            bx = gx + si * (bw + 2)
            colour = SERIES[si][0]
            if v is None:
                out.append(f'<text x="{bx + bw / 2:.1f}" y="{base - 6}" font-size="10" text-anchor="middle" fill="{MUTED}">n/a</text>')
                continue
            bh = v / ymax * h
            top = base - bh
            out.append(f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bw}" height="{bh:.1f}" rx="4" fill="{colour}"/>')
            out.append(f'<rect x="{bx:.1f}" y="{base - 6:.1f}" width="{bw}" height="6" fill="{colour}"/>')  # flat at baseline
            out.append(f'<text x="{bx + bw / 2:.1f}" y="{top - 5:.1f}" font-size="10" font-weight="600" text-anchor="middle" fill="{INK}">{fmt(v)}</text>')
        date, what, ov = MILESTONES[gi]
        cx = x0 + gi * gw + gw / 2
        out.append(f'<text x="{cx:.1f}" y="{base + 17}" font-size="11.5" font-weight="600" text-anchor="middle" fill="{INK}">{date} {what}</text>')
        out.append(f'<text x="{cx:.1f}" y="{base + 32}" font-size="10.5" text-anchor="middle" fill="{MUTED}">{ov}</text>')
    return out


w, h = 1000, 640
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">',
     f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
     f'<text x="24" y="34" font-size="20" font-weight="600" fill="{INK}">How the numbers moved, 2026-10-08 to 2026-10-30</text>',
     f'<text x="24" y="56" font-size="13" fill="{MUTED}">Bonsai 2 27B PQ2_0 on Arc B390. Decode measured with -n alone everywhere (10-08 ggml-openvino decode = clean re-measure of that code).</text>']
for i, (c, label) in enumerate(SERIES):
    x = 24 + i * 250
    s.append(f'<rect x="{x}" y="72" width="14" height="14" rx="2" fill="{c}"/>')
    s.append(f'<text x="{x + 20}" y="84" font-size="11.5" fill="{INK}">{label}</text>')

s += panel(70, 140, 880, 150, "Decode, 128 tokens  (t/s)", DECODE, 16.0, [0, 4, 8, 12, 16], lambda v: f"{v:g}")
s += panel(70, 380, 880, 150, "Prefill, 512-token prompt  (t/s)", PP512, 760.0, [0, 200, 400, 600], lambda v: f"{v:g}")

y = 578
s.append(f'<rect x="24" y="{y}" width="952" height="50" rx="6" fill="{PANEL}"/>')
s.append(f'<text x="40" y="{y + 20}" font-size="12" fill="{INK}">Native reference: paged_bench_llm_27b at 10-17, OpenVINO GenAI (VLMPipeline) at 10-30. Vulkan at 10-08 ran on Mesa 26.1.3,</text>')
s.append(f'<text x="40" y="{y + 38}" font-size="12" fill="{INK}">later on 26.2.4 with the same binary - that driver upgrade, not code, is the Vulkan step. "n/a" = not measured at that milestone.</text>')
s.append("</svg>")

out = pathlib.Path(__file__).parent / "14-results-evolution.svg"
out.write_text("\n".join(s))
print("wrote", out.name)
