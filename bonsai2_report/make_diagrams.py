#!/usr/bin/env python3
"""Generates every SVG diagram of BONSAI2_TERNARY_FULL_REPORT.md into ./diagrams.
All numbers are measured values copied from the project reports/handoffs; the two compile
timelines (d26) are drawn from data/series_{before,after}.txt (memprobe/phases.py output)."""
import os, re
from xml.sax.saxutils import escape

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagrams")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

INK, MUTED, GRID = "#1b1f24", "#6b7480", "#d9dde2"
C = {  # stroke, fill
    "blue": ("#2f6fb3", "#dce9f5"), "teal": ("#2a8c82", "#e4f1ef"), "orange": ("#c96a1b", "#fbeadb"),
    "red": ("#b8433a", "#f6dedb"), "purple": ("#6b4fa0", "#ebe5f4"), "gray": ("#6b7480", "#f1f3f5"),
    "green": ("#3a8a3a", "#e2f0e2"), "yellow": ("#a07a12", "#fbf1d3"),
}


class Svg:
    def __init__(self, w, h, title, subtitle=None):
        self.w, self.h = w, h
        self.p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
                  f'font-family="Segoe UI, Helvetica, Arial, sans-serif">',
                  '<defs>' + "".join(
                      f'<marker id="ah-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                      f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{v[0]}"/></marker>'
                      for k, v in list(C.items()) + [("ink", (INK, ""))]) + '</defs>',
                  f'<rect width="{w}" height="{h}" fill="#ffffff"/>']
        self.text(24, 34, title, 20, weight=600)
        if subtitle:
            self.text(24, 56, subtitle, 13, color=MUTED)

    def text(self, x, y, s, size=13, color=INK, weight=400, anchor="start", italic=False):
        st = ' font-style="italic"' if italic else ""
        self.p.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" '
                      f'text-anchor="{anchor}"{st}>{escape(str(s))}</text>')

    def lines(self, x, y, lines, size=12, color=INK, anchor="start", lh=None, weight=400):
        lh = lh or size * 1.35
        for i, l in enumerate(lines):
            w = weight
            if l.startswith("**"):
                l, w = l[2:], 600
            self.text(x, y + i * lh, l, size, color, w, anchor)

    def box(self, x, y, w, h, lines=(), col="blue", size=12, title=None, rx=6, dash=False, align="middle", sw=1.5):
        s, f = C[col]
        d = ' stroke-dasharray="6 4"' if dash else ""
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{f}" stroke="{s}" stroke-width="{sw}"{d}/>')
        lines = list(lines)
        n = len(lines) + (1 if title else 0)
        lh = size * 1.38
        y0 = y + h / 2 - (n - 1) * lh / 2 + size * 0.36
        tx = x + w / 2 if align == "middle" else x + 10
        if title:
            self.text(tx, y0, title, size + 1, s, 600, align)
            y0 += lh
        for i, l in enumerate(lines):
            w_ = 400
            if l.startswith("**"):
                l, w_ = l[2:], 600
            self.text(tx, y0 + i * lh, l, size, INK, w_, align)

    def rect(self, x, y, w, h, fill, stroke="none", rx=0, op=1.0):
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" opacity="{op}"/>')

    def head(self, xa, ya, x2, y2, s, size=9):
        # explicit triangle arrowhead at (x2, y2) pointing away from (xa, ya); renders everywhere
        import math
        a = math.atan2(y2 - ya, x2 - xa)
        p1 = (x2 - size * math.cos(a) + size * 0.5 * math.sin(a), y2 - size * math.sin(a) - size * 0.5 * math.cos(a))
        p2 = (x2 - size * math.cos(a) - size * 0.5 * math.sin(a), y2 - size * math.sin(a) + size * 0.5 * math.cos(a))
        self.p.append(f'<polygon points="{x2:.1f},{y2:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{s}"/>')

    def arrow(self, x1, y1, x2, y2, col="ink", label=None, dash=False, lsize=11, loff=-6, sw=1.6):
        s = INK if col == "ink" else C[col][0]
        d = ' stroke-dasharray="5 4"' if dash else ""
        self.p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{s}" stroke-width="{sw}"{d}/>')
        self.head(x1, y1, x2, y2, s)
        if label:
            self.text((x1 + x2) / 2, (y1 + y2) / 2 + loff, label, lsize, s if col != "ink" else MUTED, 400, "middle")

    def path(self, d, col="ink", sw=1.6, dash=False, arrow=False, fill="none"):
        s = INK if col == "ink" else C[col][0]
        da = ' stroke-dasharray="5 4"' if dash else ""
        self.p.append(f'<path d="{d}" stroke="{s}" stroke-width="{sw}" fill="{fill}"{da}/>')
        if arrow:
            nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", d)]
            self.head(nums[-4], nums[-3], nums[-2], nums[-1], s)

    def line(self, x1, y1, x2, y2, color=GRID, sw=1, dash=False):
        d = ' stroke-dasharray="4 4"' if dash else ""
        self.p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>')

    def circle(self, cx, cy, r, col="blue", filled=True):
        s, f = C[col]
        self.p.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{s if filled else f}" stroke="{s}" stroke-width="1.5"/>')

    def note(self, x, y, w, lines, col="yellow", size=12):
        h = len(lines) * size * 1.4 + 14
        self.box(x, y, w, h, lines, col, size, align="start")
        return h

    def save(self, name):
        self.p.append("</svg>")
        open(os.path.join(OUT, name), "w").write("\n".join(self.p))


def hbars(s, x, y, rows, vmax, width, bh=24, gap=10, unit="", label_w=230, fmt="{:.1f}", ticks=None):
    """rows: (label, value, col, note). Returns y after the last row."""
    if ticks:
        for t in ticks:
            tx = x + label_w + width * t / vmax
            s.line(tx, y - 6, tx, y + len(rows) * (bh + gap), GRID, 1, True)
            s.text(tx, y - 10, f"{t:g}", 10, MUTED, anchor="middle")
    for i, (lab, v, col, note) in enumerate(rows):
        yy = y + i * (bh + gap)
        s.text(x + label_w - 10, yy + bh * 0.68, lab, 12, INK, 400, "end")
        w = max(2, width * v / vmax)
        s.rect(x + label_w, yy, w, bh, C[col][0], rx=3)
        s.text(x + label_w + w + 6, yy + bh * 0.68, (fmt.format(v) + unit) + (f"   {note}" if note else ""), 12, INK, 600 if not note else 400)
    return y + len(rows) * (bh + gap)


# ---------------------------------------------------------------------------------------------
def d01_timeline():
    s = Svg(1100, 560, "The project at a glance - six steps from 'does not load' to 'fits in 16 GB'",
            "Intel Arc B390 iGPU, Bonsai 2 27B ternary (PQ2_0). Each step answered one question and fed the next.")
    steps = [
        ("Oct 7-8", "1. Make it run", ["4 fixes in ggml-openvino", "correct output", "prefill 3.2x Vulkan"], "blue"),
        ("Oct 8-16", "2. Native reference", ["build the OpenVINO IR", "native OV decodes 15.0", "-> 1.22x headroom"], "purple"),
        ("Oct 16-17", "3. Decode gap", ["GPU 100% busy, not BW", "extra graph ops", "12.24 -> 13.46 t/s"], "teal"),
        ("Oct 30", "4. Master + GenAI", ["PR merged upstream", "real GenAI pipeline", "final perf table"], "green"),
        ("Nov 4-5", "5. Memory", ["load peak 28-35 GiB", "3 GPU-plugin causes", "-> 10.0 GiB"], "orange"),
        ("Nov 9", "6. 16 GB test", ["15.4 GiB of RAM", "correct, 12.8 t/s", "PASS"], "red"),
    ]
    x0, dx, yl = 70, 175, 170
    s.line(50, yl, 1060, yl, MUTED, 2)
    for i, (date, name, lines, col) in enumerate(steps):
        cx = x0 + 30 + i * dx
        s.circle(cx, yl, 9, col)
        s.text(cx, yl - 22, date, 13, MUTED, 600, "middle")
        s.box(cx - 78, yl + 34, 156, 118, lines, col, 12, title=name)
        s.line(cx, yl + 9, cx, yl + 34, C[col][0], 1.5)
    s.text(550, 380, "What the two big numbers mean", 15, INK, 600, "middle")
    s.box(110, 400, 400, 120, ["ggml-openvino: 12.24 -> 13.28 t/s (final stack)", "prefill 536 t/s = 3.18x Vulkan",
                               "native OpenVINO GenAI ceiling: 14.57 t/s", "remaining gap 1.10x, root-caused"],
          "teal", 13, title="Speed")
    s.box(590, 400, 400, 120, ["peak while loading: 28.2 -> 10.0 GiB", "compile time: ~63 s -> ~10 s",
                               "runs on 15.4 GiB of physical RAM", "output unchanged, decode unchanged"],
          "orange", 13, title="Memory")
    s.save("d01-timeline.svg")


def d02_stack():
    s = Svg(1100, 640, "The software stack - who does what",
            "Three ways to run the same model on the same GPU. This project works on the middle column.")
    # columns
    s.box(40, 80, 300, 56, ["llama-cli / llama-bench"], "gray", 13)
    s.box(400, 80, 300, 56, ["llama-cli / llama-bench"], "gray", 13)
    s.box(760, 80, 300, 56, ["llm_bench.py (Python)"], "gray", 13)
    s.box(40, 160, 660, 74, ["PrismML fork of llama.cpp: reads the GGUF file (knows the PQ2_0 type),",
                             "builds a ggml compute graph for every batch of tokens"], "blue", 13, title="llama.cpp")
    s.box(40, 258, 300, 74, ["SPIR-V shaders via Mesa", "(baseline for comparison)"], "gray", 12, title="ggml-vulkan")
    s.box(400, 258, 300, 74, ["translates the ggml graph", "into an OpenVINO model"], "blue", 12, title="ggml-openvino backend")
    s.box(760, 160, 300, 172, ["reads an OpenVINO IR file", "(xml graph + bin weights)", "tokenizer, sampling, KV cache",
                               "", "the 'native' reference:", "no ggml in the loop"], "purple", 12, title="OpenVINO GenAI")
    s.box(400, 356, 660, 92, ["graph transformations (fusions, weight decompression patterns), kernel selection,",
                              "constant upload to GPU memory, OpenCL / oneDNN kernel compilation",
                              "TernOCL = the 2-bit ternary matmul kernels from OpenVINO PR #38460"], "teal", 12,
          title="OpenVINO runtime + Intel GPU plugin")
    s.box(40, 356, 300, 92, ["Mesa ANV driver"], "gray", 12)
    s.box(40, 476, 1020, 82, ["Intel Arc B390 integrated GPU (Xe3). No VRAM of its own: it uses system RAM.",
                              "LPDDR5X-9600, 8 x 16-bit channels = 153.6 GB/s peak, 62 GB shared with the CPU on this box"],
          "orange", 13, title="Hardware")
    for x in (190, 550):
        s.arrow(x, 136, x, 160)
    s.arrow(910, 136, 910, 160)
    s.arrow(190, 234, 190, 258)
    s.arrow(550, 234, 550, 258)
    s.arrow(550, 332, 550, 356)
    s.arrow(910, 332, 910, 356)
    s.arrow(190, 332, 190, 356)
    s.arrow(190, 448, 190, 476)
    s.arrow(730, 448, 730, 476)
    s.text(550, 592, "Same GPU, same weights, three paths: comparing them separates 'the GPU/kernels are slow' from 'the integration is slow'.",
           13, MUTED, 400, "middle")
    s.save("d02-stack.svg")


def d03_architecture():
    s = Svg(1100, 620, "Bonsai 2 27B - a hybrid-attention model (GGUF architecture 'qwen35')",
            "27 B parameters, dense (no experts). 64 layers, hidden size 5120, vocabulary 248 320, context up to 262 k tokens.")
    s.box(40, 90, 150, 60, ["token embedding", "248320 x 5120"], "gray", 12)
    s.arrow(190, 120, 222, 120)
    # 16 repeats
    x = 226
    for g in range(4):
        for k in range(4):
            col = "teal" if k < 3 else "purple"
            s.rect(x, 96, 38, 48, C[col][1], C[col][0], 4)
            s.text(x + 19, 125, "GDN" if k < 3 else "ATT", 11, C[col][0], 600, "middle")
            x += 42
        x += 10
    s.text(560, 170, "the pattern [GDN, GDN, GDN, Attention] repeats 16 times = 48 GatedDeltaNet + 16 attention layers",
           12, MUTED, 400, "middle")
    s.arrow(x - 4, 120, x + 26, 120)
    s.box(x + 28, 90, 110, 60, ["LM head", "5120 -> 248320"], "gray", 12)
    # detail boxes
    s.box(40, 210, 500, 230, [
        "Linear attention: no attention matrix at all.",
        "Keeps a fixed-size memory per head - the recurrent state,",
        "a 128 x 128 matrix - and updates it as each token streams by.",
        "",
        "**cost per token is constant, however long the context is",
        "",
        "16 q/k heads serve 48 value heads (grouped)",
        "pieces: 1-D conv over q/k/v, L2-normalize q and k,",
        "two learned gates (alpha, beta), the recurrent update",
        "state per layer: [1, 48, 128, 128] f32 = 3.1 MB"], "teal", 12, title="GatedDeltaNet layer (48 of them)")
    s.box(560, 210, 500, 230, [
        "Ordinary softmax attention, so the model can still",
        "do exact long-range look-ups.",
        "",
        "24 query heads over 4 key/value heads (GQA)",
        "",
        "**keeps a KV cache that grows with the context",
        "",
        "only 1 layer in 4, so the KV cache is small"], "purple", 12, title="Full-attention layer (16 of them)")
    s.box(40, 460, 1020, 70, ["every layer, of both kinds, ends with the same SwiGLU MLP:  5120 -> 17408 (gate) and 17408 (up) -> SiLU(gate) x up -> 5120 (down)",
                              "these big matrices hold most of the 27 B parameters, and they are the ternary ones"], "blue", 12,
          title="Feed-forward (SwiGLU)")
    s.note(40, 548, 1020, ["Why it matters here: GatedDeltaNet has a more intricate graph than attention (conv, per-head norm, gates, a scan), so most shape bugs",
                           "and the biggest decode overhead were found there. The small alpha/beta projections are stored in BF16, not ternary."])
    s.save("d03-architecture.svg")


def d04_sizes():
    s = Svg(1000, 430, "Why ternary: the same 26.9 B parameters at different precisions",
            "Approximate weight size = parameters x bits per weight / 8. Only the ternary row was measured (6.70 GiB file).")
    rows = [("fp16 (16 bpw)", 53.8, "gray", ""), ("int8 (~8.5 bpw)", 28.6, "gray", ""),
            ("int4 (~4.5 bpw)", 15.1, "blue", ""), ("ternary PQ2_0 (2.13 bpw)", 7.2, "teal", "measured file: 7.2 GB")]
    y = hbars(s, 30, 110, rows, 60, 600, bh=34, gap=18, unit=" GB", ticks=[0, 32, 48])
    x16 = 30 + 230 + 600 * 16 / 60
    s.line(x16, 96, x16, y, C["red"][0], 2, True)
    s.text(x16 + 6, 100, "16 GB = all of a laptop's RAM", 12, C["red"][0], 600)
    s.note(30, 330, 940, ["A 27 B model normally needs a workstation. At ~2 bits per weight it fits next to the OS on a 16 GB laptop - if",
                          "the runtime does not need extra copies of the weights while loading. Step 5 of this report is about exactly that."])
    s.save("d04-sizes.svg")


def d05_pq2():
    s = Svg(1000, 560, "PQ2_0 - how one block of 128 ternary weights is stored",
            "'Prism Q2_0', a ggml type that exists only in the PrismML fork of llama.cpp (type id 142).")
    s.text(24, 96, "One block = 128 weights = 34 bytes", 14, INK, 600)
    s.box(24, 108, 90, 50, ["d", "f16 scale"], "blue", 12)
    s.box(118, 108, 640, 50, ["qs[32]  -  32 bytes", "128 codes x 2 bits, 4 codes per byte"], "teal", 12)
    s.lines(772, 126, ["2 + 32 = 34 bytes", "**= 2.125 bits per weight"], 13)
    s.text(24, 200, "Inside one byte - code j sits at bits [2j, 2j+1], lowest bits first", 14, INK, 600)
    for i, lab in enumerate(["bits 7..6", "bits 5..4", "bits 3..2", "bits 1..0"]):
        s.box(24 + i * 110, 212, 110, 54, [lab, f"code {3 - i}"], "gray", 12, rx=0)
    s.text(500, 234, "the same rule OpenVINO uses for its 'u2' type", 12, MUTED)
    s.text(500, 252, "-> converting PQ2_0 to OpenVINO is a plain memcpy (section 5)", 12, MUTED)
    s.text(24, 310, "From code to weight:   weight = (code - 1) x d", 14, INK, 600)
    for i, (c, v, col) in enumerate([("0", "-d", "red"), ("1", "0", "gray"), ("2", "+d", "green"), ("3", "never used", "yellow")]):
        s.box(24 + i * 175, 326, 160, 56, [f"code {c}", f"-> {v}"], col, 13)
    s.note(24, 410, 950, ["The weights really are -1, 0 or +1 (times one scale per 128 weights) - the model was trained/quantized that way.",
                          "log2(3) = 1.58 bits would be the theoretical minimum. Wasting the 4th code buys unpacking with a shift and a mask,",
                          "no division. The sibling type PTQ1_0 packs 5 trits per byte (1.75 bpw) at the cost of divisions to unpack."])
    s.save("d05-pq2_0-block.svg")


def d06_hadamard():
    s = Svg(1100, 600, "The twist: the weights live in a rotated basis (Hadamard rotation)",
            "Why the model needs an extra transform in front of every ternary matmul, and why it must not be skipped.")

    def dots(x0, y0, vals, col, label):
        s.text(x0, y0 - 12, label, 13, INK, 600)
        s.line(x0, y0 + 60, x0 + 420, y0 + 60, GRID)
        for i, v in enumerate(vals):
            x = x0 + 8 + i * 13
            s.line(x, y0 + 60, x, y0 + 60 - v * 50, C[col][0], 5)
    import math
    raw = [0.08 * math.sin(i * 1.7) for i in range(32)]
    raw[11] = 1.0
    rot = [0.35 * math.sin(i * 2.3 + 0.5) + 0.1 * math.cos(i * 0.9) for i in range(32)]
    dots(40, 100, raw, "red", "Before: one group of weights with an outlier")
    dots(600, 100, rot, "green", "After rotation: the outlier is spread over all 1024 inputs")
    s.lines(40, 200, ["Ternary has 3 levels. The scale is set by the outlier,", "so every small weight rounds to 0: information lost."], 12, MUTED)
    s.lines(600, 200, ["Values are now roughly Gaussian and similar in size,", "so -1/0/+1 x scale keeps most of the information."], 12, MUTED)
    s.arrow(470, 140, 590, 140, "green", "x H (offline)")
    s.box(40, 260, 1020, 96, ["W_rot = W . H         done once, offline, when the model was quantized",
                              "x_rot = H . x          must happen at run time, on every forward pass",
                              "W_rot . x_rot = W . x  because H is orthogonal - the result is unchanged"], "blue", 14,
          title="The math (H = 1024 x 1024 normalized Walsh-Hadamard matrix, entries +-1/32, plus a fixed sign flip)")
    s.text(40, 392, "In the ggml graph, in front of every ternary matmul:", 13, INK, 600)
    chain = [("x", "gray"), ("MUL by signs", "blue"), ("RESHAPE", "blue"), ("MUL_MAT by H", "blue"), ("RESHAPE back", "blue"), ("MUL_MAT by W", "teal")]
    for i, (t, col) in enumerate(chain):
        bx = 40 + i * 172
        s.box(bx, 406, 150, 44, [t], col, 12)
        if i:
            s.arrow(bx - 22, 428, bx, 428)
    s.text(380, 470, "[T, K] -> [T*K/1024, 1024]", 11, MUTED, 400, "middle")
    s.note(40, 490, 1020, ["Skip the activation rotation and the output is garbage, not 'slightly worse'. So the rotation is part of the model.",
                           "Ordinary ops (MUL, RESHAPE, MUL_MAT) - any backend can run it, but ~400 extra 1024x1024 matmuls per token is slow.",
                           "OpenVINO's GPU plugin (PR #38460) recognises the chain and folds it into the ternary matmul kernel: FuseHadamardIntoFC."])
    s.save("d06-hadamard.svg")


def d07_ggml_ov_flow():
    s = Svg(1100, 470, "How the OpenVINO backend runs a llama.cpp model",
            "Compile once per graph shape, then replay it for every batch of tokens. The compile step is where the memory story (step 5) happens.")
    boxes = [("llama.cpp", ["builds a ggml graph", "for this batch"], "blue"),
             ("ggml-openvino", ["translates every ggml op", "weights -> Constants", "token dim T -> dynamic"], "blue"),
             ("core.compile_model", ["GPU plugin: fusions,", "kernel choice, upload", "weights to GPU memory"], "teal"),
             ("infer", ["runs the compiled model", "on the GPU"], "green")]
    for i, (t, l, col) in enumerate(boxes):
        x = 40 + i * 265
        s.box(x, 100, 225, 110, l, col, 12, title=t)
        if i:
            s.arrow(x - 40, 155, x, 155)
    s.path("M 917 210 C 917 290, 330 290, 330 214", "green", dash=True, arrow=True)
    s.text(620, 290, "next batch with the same structure: reuse the compiled model (cache hit), skip compile", 12, C["green"][0], 400, "middle")
    s.box(40, 320, 500, 120, ["one compiled model must serve T = 42 (prompt),", "T = 2 (warm-up), T = 1 (decode), ...",
                              "so no shape constant in the graph may depend on T", "-> the reshape bug of fix 3"], "orange", 12,
          title="The dynamic token dimension")
    s.box(560, 320, 500, 120, ["~10-60 s for this 27 B model (tens of thousands of ops,", "kernel builds, weight upload/repacking)",
                               "llama-bench --no-warmup puts it INSIDE the timing:", "pp128 reads 4 t/s instead of 234"], "red", 12,
          title="Compile is expensive")
    s.save("d07-ggml-openvino-flow.svg")


def d08_pq2_to_u2():
    s = Svg(1100, 430, "Fix 1 - reading PQ2_0 as OpenVINO u2: a memcpy, not a conversion",
            "OpenVINO has no signed 2-bit type, but 'u2 with zero point 1' is exactly PQ2_0's (code - 1) x d.")
    s.text(40, 96, "ggml tensor in the GGUF file: scale and codes interleaved per block", 13, INK, 600)
    for i in range(4):
        x = 40 + i * 255
        s.box(x, 108, 50, 40, ["d" + str(i)], "blue", 12, rx=2)
        s.box(x + 50, 108, 195, 40, [f"qs{i} (32 B)"], "teal", 12, rx=2)
    s.text(1065, 133, "...", 16, MUTED)
    s.text(40, 214, "OpenVINO model: three separate Constants", 13, INK, 600)
    s.box(40, 226, 760, 44, ["u2 codes: qs0 | qs1 | qs2 | qs3 | ...   (byte-for-byte copy, no bit shuffling)"], "teal", 12, rx=2)
    s.box(820, 226, 240, 44, ["f16 scales: d0 d1 d2 d3 ..."], "blue", 12, rx=2)
    s.box(40, 290, 300, 44, ["zero point: one f16 Constant = 1"], "gray", 12, rx=2)
    for i in range(4):
        x = 40 + i * 255
        s.arrow(x + 147, 148, 60 + i * 190 + 80, 226, "teal")
        s.arrow(x + 25, 148, 850 + i * 50, 226, "blue", dash=True)
    s.note(40, 356, 1020, ["Both formats put element j at bits [2(j%4), 2(j%4)+1] of byte j/4, lowest bits first (OpenVINO packs every sub-byte type",
                           "LSB-first except u1). Only the scales need gathering into their own array, because ggml interleaves them with the codes."])
    s.save("d08-pq2_0-to-u2.svg")


def d09_decompression():
    s = Svg(1100, 470, "Fix 1 + fix 4 - the weight-decompression pattern the GPU plugin looks for",
            "The backend emits ordinary ops; the plugin recognises the pattern and replaces it with one compressed matmul.")
    chain = [("Constant", "u2 codes", "teal"), ("Convert", "-> f16", "blue"), ("Subtract", "zero point 1", "blue"),
             ("Multiply", "f16 scales", "blue"), ("Reshape", "[N,K/128,128]->[N,K]", "blue"), ("MatMul", "with activations", "purple")]
    for i, (t, sub, col) in enumerate(chain):
        x = 30 + i * 177
        s.box(x, 96, 157, 62, [sub], col, 12, title=t)
        if i:
            s.arrow(x - 20, 127, x, 127)
    s.arrow(550, 168, 550, 212, "teal", "GPU plugin pattern match", loff=0)
    s.box(250, 216, 600, 62, ["reads 2-bit codes straight from memory, never materializes f16 weights", "-> TernOCL kernel when it is a ternary u2 weight"],
          "teal", 12, title="FullyConnectedCompressed (one op)")
    s.box(30, 304, 510, 132, ["The zero point must be a Constant, not Convert(int Constant).",
                              "The ternary kernel reads the value directly when it builds.",
                              "A Convert has no value yet at that point, so the fusion",
                              "silently does not happen and the slow path runs.",
                              "(the same trap as an earlier gemma-4 MoE bug)"], "red", 12, title="Trap 1")
    s.box(560, 304, 510, 132, ["The backend forced token_embd and output weights to 8-bit",
                               "for every model ('requantization'). On a ternary model that",
                               "inflates 2.13 bpw to 8 bpw AND moves the output projection",
                               "off the ternary path. Fix 4: never requantize PQ2_0."], "red", 12, title="Trap 2 = fix 4")
    s.save("d09-decompression-pattern.svg")


def d10_gdn_fix():
    s = Svg(1100, 520, "Fix 2 - the joint q/k normalization in GatedDeltaNet",
            "Bonsai normalizes q and k together as one 32-head tensor, then hands the op two 16-head views of it.")
    s.box(40, 90, 460, 56, ["L2-normalize [q (16 heads) | k (16 heads)] = 32 heads at once"], "blue", 12)
    s.box(40, 170, 220, 50, ["VIEW q = heads 0..15"], "gray", 12)
    s.box(280, 170, 220, 50, ["VIEW k = heads 16..31"], "gray", 12)
    s.arrow(150, 146, 150, 170)
    s.arrow(390, 146, 390, 170)
    s.text(40, 254, "Before: VIEW is a pass-through on this backend ('the consumer slices').", 13, C["red"][0], 600)
    s.text(40, 274, "The GDN translator sliced its v input - but not q and k.", 13, C["red"][0], 600)
    s.box(40, 290, 460, 50, ["q and k each arrive with all 32 heads"], "red", 12)
    s.box(40, 356, 460, 66, ["grouped heads: tile x3 (16 q/k heads -> 48 v heads)", "x3 computed from declared 16, applied to actual 32 -> 96"], "red", 12)
    s.box(40, 438, 460, 44, ["96 q/k heads vs 48 v heads -> op rejects the graph"], "red", 12)
    s.arrow(270, 340, 270, 356, "red")
    s.arrow(270, 422, 270, 438, "red")
    s.text(600, 254, "After: re-slice q and k on the head axis, like v.", 13, C["green"][0], 600)
    s.box(600, 290, 460, 50, ["q = 16 heads, k = 16 heads"], "green", 12)
    s.box(600, 356, 460, 66, ["tile x3 -> 48 heads", "matches 48 value heads"], "green", 12)
    s.box(600, 438, 460, 44, ["correct output"], "green", 12)
    s.arrow(830, 340, 830, 356, "green")
    s.arrow(830, 422, 830, 438, "green")
    s.note(600, 90, 460, ["Why nobody hit it before: Qwen3.5 / 3.6 MoE normalize", "q and k separately. Here the declared shape said 16",
                          "and the real tensor was 32 - compare declared vs actual.", "No-op when they match, so other models are untouched."])
    s.save("d10-fix-gdn.svg")


def d11_reshape_fix():
    s = Svg(1100, 500, "Fix 3 - a reshape that baked in the token count",
            "The Hadamard rotation reshapes [T, 5120] -> [T*5, 1024]. The token axis is split, not kept.")
    s.box(40, 90, 300, 56, ["input  [1, 1, T, 5120]", "T is dynamic"], "blue", 13)
    s.box(40, 180, 300, 56, ["output [1, 1, T*5, 1024]"], "blue", 13)
    s.arrow(190, 146, 190, 180)
    s.text(200, 168, "RESHAPE", 11, MUTED)
    s.text(380, 104, "Old rule: 'the output axis with the same extent as the input's dynamic axis gets -1'", 13, INK, 600)
    s.lines(380, 128, ["no output axis has extent T (it has T*5) -> rule finds nothing",
                       "-> whole target shape written as constants, from the first graph captured:",
                       "    warm-up with T = 2 -> target [1, 1, 10, 1024]",
                       "-> replay with a 42-token prompt: [1,1,42,5120] cannot become [1,1,10,1024]"], 12, C["red"][0])
    s.box(380, 230, 680, 40, ["error: Requested output shape [1,1,10,1024] is incompatible with input shape [1,1,42,5120]"], "red", 12)
    s.text(380, 312, "New fallback rule (only when the old one fails): 'the output axis that spans the same elements'", 13, INK, 600)
    s.lines(380, 336, ["everything above it is static, and the element count up to and including it matches",
                       "-> axis 2 is the dynamic one -> target [1, 1, -1, 1024] -> works for every T"], 12, C["green"][0])
    s.box(380, 382, 680, 40, ["target [1, 1, -1, 1024]   (computed at run time)"], "green", 12)
    s.note(40, 270, 300, ["Nothing about this is ternary-", "specific: any reshape that splits or", "merges the dynamic axis had this",
                          "latent bug. It is the one fix that", "could go upstream on its own."])
    s.save("d11-fix-reshape.svg")


def d12_cascade():
    s = Svg(1100, 460, "The debugging lesson - the error message was five steps away from the cause",
            "Fix 3 never showed up as a shape error. It surfaced as a kernel-acceptance failure.")
    steps = [("reshape target", "baked as constants", "red"), ("shape becomes", "static", "orange"),
             ("gate/up merge pass", "requires dynamic -> skips", "orange"), ("ffn_gate keeps 3", "fused ops", "orange"),
             ("TernOCL epilogue", "table knows 1-2 -> reject", "orange"), ("hard build failure", "(by design)", "red")]
    for i, (a, b, col) in enumerate(steps):
        x = 30 + i * 177
        s.box(x, 110, 157, 70, [a, b], col, 12)
        if i:
            s.arrow(x - 20, 145, x, 145)
    s.text(108, 216, "real cause", 13, C["red"][0], 600, "middle")
    s.text(993, 216, "what the log said", 13, C["red"][0], 600, "middle")
    s.box(560, 236, 510, 60, ["'ffn_gate carries a Hadamard input transform but the", "TernOCL impl rejected it: fused chain has no folded epilogue'"], "gray", 11)
    s.box(30, 236, 500, 60, ["Why a reject is fatal here: this matmul has the rotation folded in,", "and TernOCL is the only kernel that applies it. Falling back = garbage."], "gray", 11)
    s.note(30, 320, 1040, ["Rule of thumb: when a GPU-plugin fusion mysteriously does not fire, check whether shapes are still dynamic BEFORE studying the",
                           "pattern matcher. Several passes silently require dynamic shapes because they target generative models. After fix 3 the merged",
                           "nodes appear as ffn_gate_fused_2FCs, z_fused_2FCs, Qcur_full_fused_3FCs - the '_fused_NFCs' suffix is the merge naming its result."])
    s.save("d12-fix-chain.svg")


def d13_dispatch():
    s = Svg(1100, 470, "TernOCL - how one compiled model uses different kernels for prompt and decode",
            "The kernel is chosen at run time from M = the number of tokens in this call, not at compile time.")
    s.box(40, 100, 220, 70, ["FullyConnected", "(ternary weights)"], "teal", 13)
    s.box(320, 100, 230, 70, ["M = tokens in this call", "launch class = bucket(M), 8 slots"], "gray", 12)
    s.arrow(260, 135, 320, 135)
    s.box(640, 80, 420, 70, ["GEMV: tuned for 1 or a few tokens", "reads every weight once per token -> memory-bound"], "blue", 12, title="M <= 8  (decode)")
    s.box(640, 170, 420, 86, ["tiled GEMM, DPAS matrix instructions", "default: 2-bit codes -> fp16 in registers, fp16 x fp16",
                              "INT8_PREFILL=1: int2 x int8 DPAS, activations -> int8"], "purple", 12, title="M > 8  (prompt / batch)")
    s.arrow(550, 125, 640, 115)
    s.arrow(550, 145, 640, 210)
    s.text(40, 230, "Each kernel is built with clBuildProgram on first use", 12, MUTED)
    s.text(40, 248, "and cached per launch class, so a node collects a", 12, MUTED)
    s.text(40, 266, "handful of OpenCL kernels over the process lifetime.", 12, MUTED)
    s.box(40, 300, 500, 140, ["2-bit weights unpacked to fp16 in registers", "activations stay fp16 - no extra rounding",
                              "pp128 237 t/s, pp512 536 t/s"], "blue", 12, title="Default prefill")
    s.box(560, 300, 500, 140, ["weights never unpacked; activations quantized to int8", "per (row, 128 group) - faster, slightly lossy",
                               "pp128 547 t/s (2.3x), pp512 571 t/s (1.06x)", "off by default; env OV_GPU_TERNOCL_INT2_INT8_PREFILL=1"],
          "purple", 12, title="int8 prefill (opt-in)")
    s.save("d13-kernel-dispatch.svg")


def d14_phase1_results():
    s = Svg(1100, 460, "Step 1 result (2026-10-08): it runs, and the ternary kernels are not optional",
            "Arc B390 iGPU, llama-bench -r 3 -ngl 99, tokens/second. OpenVINO PR #38460 build 5c6d4692ee.")
    s.text(40, 96, "Prompt processing, 512 tokens (pp512)", 14, INK, 600)
    hbars(s, 40, 112, [("ggml-openvino + TernOCL", 527.4, "teal", ""), ("ggml-openvino, int8 prefill", 571.1, "purple", ""),
                       ("ggml-vulkan", 165.4, "gray", "")], 600, 260, label_w=210, fmt="{:.0f}")
    s.text(600, 96, "Generation (tg32)", 14, INK, 600)
    hbars(s, 600, 112, [("ggml-openvino + TernOCL", 12.25, "teal", ""), ("ggml-vulkan", 12.32, "gray", ""),
                        ("TernOCL disabled", 1.24, "red", "9.9x slower")], 14, 200, label_w=190, fmt="{:.2f}")
    s.box(40, 250, 500, 92, ["Vulkan is flat from 128 to 512 tokens (161 -> 165): a per-token kernel.",
                             "OpenVINO scales (234 -> 527): a tiled GEMM that reuses weights", "across tokens. 3.2x faster at 512 tokens."], "teal", 12, title="Prefill: 3.2x Vulkan")
    s.box(600, 250, 460, 92, ["Without the PR's kernels the u2 weights take the generic", "decompression path: 1.4 t/s prefill (167x slower).",
                              "The kernels are the difference between usable and not."], "red", 12, title="TernOCL is required")
    s.note(40, 362, 1020, ["What we believed then (later shown WRONG in step 3): 'decode is a tie, pinned by an ~80 GB/s bandwidth wall'.",
                           "This machine's peak is 153.6 GB/s, and native OpenVINO decodes the same weights 22% faster. Measure the ceiling, don't assume it."], "orange")
    s.save("d14-step1-results.svg")


def d15_ir_pipeline():
    s = Svg(1100, 520, "Step 2 - building a native OpenVINO model to compare against (one-time, ~150 GB of disk)",
            "Native OpenVINO cannot read GGUF. PR #38460's converter needs a Bonsai 1 IR as a graph template.")
    s.box(40, 90, 230, 70, ["Bonsai 1 27B, unpacked", "HuggingFace, 51 GB"], "gray", 12)
    s.box(310, 90, 230, 70, ["optimum-cli export", "fp16 IR, 51 GB, ~60 GB RAM"], "gray", 12)
    s.box(580, 90, 230, 70, ["quantize_ir_ternary.py", "template IR, 6.34 GiB"], "purple", 12)
    s.arrow(270, 125, 310, 125)
    s.arrow(540, 125, 580, 125)
    s.box(580, 200, 230, 70, ["Bonsai 2 GGUF", "PQ2_0, 7.2 GB"], "teal", 12)
    s.box(850, 145, 210, 80, ["bonsai2_gguf_to_ir.py", "graph from template,", "weights from GGUF"], "purple", 12)
    s.arrow(810, 125, 850, 170)
    s.arrow(810, 235, 850, 200)
    s.box(850, 265, 210, 70, ["Bonsai 2 u2 IR", "6.9 GB + 0.34 GB embed."], "green", 12)
    s.arrow(955, 225, 955, 265)
    s.box(560, 360, 240, 70, ["paged_bench_llm_27b", "PR's own tool (later removed)"], "purple", 12)
    s.box(820, 360, 240, 70, ["OpenVINO GenAI", "VLMPipeline + llm_bench"], "purple", 12)
    s.arrow(930, 335, 680, 360)
    s.arrow(955, 335, 940, 360)
    s.box(40, 200, 500, 110, ["Run with the PR's 21-token reference prompt, the output", "token ids matched the PR's published ids byte-for-byte.",
                              "That proves the GGUF -> IR weight mapping is right AND that", "the native number is a real ceiling for these weights."],
          "green", 12, title="Validation")
    s.note(40, 340, 480, ["Traps: pip-wheel GenAI venvs do not contain TernOCL", "(you would measure the 1.4 t/s path). No prebuilt u2 IR", "exists on HF; the one 'Bonsai OpenVINO' repo is a",
                          "different 1-bit scheme with an untrusted plugin .so."])
    s.save("d15-native-ir-pipeline.svg")


def d16_bandwidth():
    s = Svg(1000, 420, "Step 3a - 'we are bandwidth-bound' was wrong",
            "Decode reads all 6.86 GB of weights once per token, so tokens/s x bytes = achieved bandwidth.")
    rows = [("theoretical peak (LPDDR5X)", 153.6, "gray", "8 x 16-bit x 9600 MT/s"),
            ("TernOCL GEMV kernels alone", 106, "teal", "same in BOTH backends (69% of peak)"),
            ("native OpenVINO, whole token", 103, "purple", "15.0 t/s"),
            ("ggml-openvino, whole token", 88, "blue", "12.25 t/s"),
            ("old 'hardware wall' claim", 80, "red", "retracted")]
    hbars(s, 30, 100, rows, 160, 430, bh=30, gap=14, unit=" GB/s", label_w=250, fmt="{:g}", ticks=[0, 40, 80, 120, 160])
    s.note(30, 330, 940, ["The ternary kernels are equally fast on both paths. The lower whole-token number for ggml-openvino means its",
                          "token takes longer for reasons other than reading weights - i.e. extra work around the GEMVs (next diagram)."])
    s.save("d16-bandwidth.svg")


def d17_decode_budget():
    s = Svg(1100, 520, "Step 3b - where one decode token's time goes",
            "GGML_OPENVINO_PROFILING=2 per-op timings in steady-state decode; GPU busy measured from the xe PMU.")
    sc = 9.5
    x0, y1, y2, h = 200, 110, 190, 46
    s.text(x0 - 12, y1 + 30, "ggml-openvino", 13, INK, 600, "end")
    s.text(x0 - 12, y2 + 30, "native OpenVINO", 13, INK, 600, "end")
    s.rect(x0, y1, 64.72 * sc, h, C["teal"][0], rx=3)
    s.text(x0 + 64.72 * sc / 2, y1 + 29, "ternary GEMVs 64.72 ms", 13, "#ffffff", 600, "middle")
    s.rect(x0 + 64.72 * sc, y1, 11.98 * sc, h, C["red"][0], rx=3)
    s.text(x0 + 76.7 * sc + 8, y1 + 29, "76.70 ms = 13.0 t/s", 13, INK, 600)
    s.text(x0 + 70.7 * sc, y1 - 8, "other ops 11.98", 12, C["red"][0], 600, "middle")
    s.rect(x0, y2, 64.72 * sc, h, C["teal"][0], rx=3)
    s.text(x0 + 64.72 * sc / 2, y2 + 29, "same GEMVs", 13, "#ffffff", 600, "middle")
    s.rect(x0 + 64.72 * sc, y2, 2.47 * sc, h, C["red"][0], rx=3)
    s.text(x0 + 67.19 * sc + 8, y2 + 29, "67.19 ms = 14.9 t/s", 13, INK, 600)
    s.text(40, 280, "The 11.98 ms of 'other' (2718 ops per token, mean 4.9 us each - count matters, not speed):", 13, INK, 600)
    rows = [("Transpose (273)", 2.80, "red", ""), ("Convert (275)", 2.52, "red", ""), ("GatedDeltaNet (48)", 1.87, "gray", ""),
            ("Result writeback (96)", 1.06, "red", ""), ("FC bf16 alpha/beta (97)", 0.90, "gray", ""), ("RMS norm (209)", 0.70, "gray", ""),
            ("StridedSlice (336)", 0.39, "gray", ""), ("rest (Concat, SDPA, Add...)", 1.74, "gray", "")]
    hbars(s, 40, 296, rows, 3.2, 420, bh=18, gap=6, unit=" ms", label_w=220, fmt="{:.2f}")
    s.box(760, 280, 300, 200, ["GPU compute engine busy:", "ggml-openvino 99.99%", "native OpenVINO 99.95%", "",
                               "-> NOT host dispatch, NOT idle,", "NOT bandwidth: ggml-openvino just", "gives the GPU more small ops"], "blue", 12, title="Ruled out")
    s.save("d17-decode-budget.svg")


def d18_gdn_state():
    s = Svg(1100, 470, "Step 3c - the biggest single item: shipping the recurrent state back and forth",
            "Three op types appeared exactly 96 times per token = 48 GDN layers x (state in + state out).")
    xs = [40, 220, 400, 580, 760, 920]
    items = [("ggml buffer", "f32 [B,H,v,k]", "gray"), ("Transpose", "{0,1,3,2}", "red"), ("GatedDeltaNet", "wants [B,H,k,v]", "teal"),
             ("Transpose", "back", "red"), ("Convert", "-> f32", "orange"), ("Result", "write back", "orange")]
    for (x, (a, b, col)) in zip(xs, items):
        s.box(x, 100, 150, 64, [b], col, 12, title=a)
    for a, b in zip(xs[:-1], xs[1:]):
        s.arrow(a + 150, 132, b, 132)
    s.path("M 995 164 C 995 230, 115 230, 115 168", "gray", dash=True, arrow=True)
    s.text(555, 226, "every token, every GDN layer: 3.1 MB per layer, ~151 MB per token", 12, MUTED, 400, "middle")
    s.box(40, 256, 500, 116, ["the buffer is only read and written by this op pair,", "so it can simply hold the op's own layout:",
                              "96 transposes -> 0, 2718 -> 2622 ops per token", "decode 13.05 -> 13.45 t/s (+3.0%), output identical"],
          "green", 12, title="Fixed: keep the state in the op's layout")
    s.box(560, 256, 500, 116, ["ggml owns the state in f32, the GPU computes in f16:", "48 Convert + 96 Result = ~3.4 ms per token",
                               "needs a device-resident state (OpenVINO Variable),", "worth ~+4.7% -> ~14.0 t/s. Hard part: llama.cpp clears/copies state slots"],
          "orange", 12, title="Not fixed yet: the dtype + writeback half")
    s.note(40, 392, 1020, ["Guarded on a square state (128 x 128 here): otherwise skipping the permute would change the buffer shape ggml allocated.",
                           "OpenVINO's own GGUF frontend made the identical change upstream (630516e048) - a good sign the idea is right."])
    s.save("d18-gdn-state.svg")


def d19_decode_waterfall():
    s = Svg(1100, 470, "Step 3 result - decode, one change at a time (same-session A/B, tg128)",
            "Four changes on top of the step-1 branch: three ported from ravi9/dev_backend_openvino, one written here.")
    vals = [("step-1 branch", 12.24, None), ("+ graft of 15 backend commits (features off)", 12.35, "+0.9%"),
            ("+ attention sizes from KQ mask (fused SDPA)", 12.68, "+2.7%"),
            ("+ 4-D stateless KV cache (KV_4D=1)", 12.86, "+1.4%"), ("+ KV slice off", 13.07, "+1.6%"),
            ("+ GDN state in op layout (our fix)", 13.46, "+3.0%")]
    base, top, x0, w = 11.5, 15.5, 400, 600

    def X(v):
        return x0 + (v - base) / (top - base) * w
    for i, (lab, v, d) in enumerate(vals):
        y = 92 + i * 46
        s.text(x0 - 12, y + 22, lab, 12, INK, 400, "end")
        prev = vals[i - 1][1] if i else base
        if i:
            s.rect(X(prev), y, X(v) - X(prev), 32, C["green"][0], rx=3)
            s.text(X(v) + 8, y + 21, f"{v:.2f}  ({d})", 12, INK, 600)
        else:
            s.rect(X(base), y, X(v) - X(base), 32, C["blue"][0], rx=3)
            s.text(X(v) + 8, y + 21, f"{v:.2f}", 12, INK, 600)
    for v, lab, col in [(15.03, "native OpenVINO 15.03", "purple"), (12.71, "Vulkan 12.71", "gray")]:
        s.line(X(v), 84, X(v), 368, C[col][0], 2, True)
        s.text(X(v), 384, lab, 12, C[col][0], 600, "middle")
    s.note(40, 400, 1020, ["+10.0% total, gap to the native ceiling 1.22x -> 1.12x. Axis starts at 11.5 t/s. Each step measured back-to-back in one",
                           "session: cross-day numbers on this box drift by several percent (a Mesa upgrade alone moved Vulkan +3%)."])
    s.save("d19-decode-waterfall.svg")


def d20_final_perf():
    s = Svg(1100, 470, "Step 4 - final speed on OpenVINO master (2026-10-30)",
            "OpenVINO master 5d07300df3 (PR #38460 merged) + GenAI 49237dab; llama.cpp synced with dev_backend_openvino. t/s.")
    s.text(40, 96, "Prefill, 512-token prompt", 14, INK, 600)
    hbars(s, 40, 112, [("OpenVINO GenAI, int8 prefill", 740.4, "purple", ""), ("OpenVINO GenAI", 680.0, "purple", ""),
                       ("ggml-openvino, int8 prefill", 570.9, "teal", ""), ("ggml-openvino", 536.1, "teal", ""),
                       ("ggml-vulkan", 168.7, "gray", "")], 800, 250, label_w=220, fmt="{:.0f}")
    s.text(600, 96, "Decode, tg128", 14, INK, 600)
    hbars(s, 600, 112, [("OpenVINO GenAI", 14.57, "purple", ""), ("ggml-openvino", 13.28, "teal", ""), ("ggml-vulkan", 12.71, "gray", "")],
          16, 230, label_w=160, fmt="{:.2f}")
    s.box(40, 300, 330, 100, ["ggml-openvino vs Vulkan:", "prefill 3.18x (3.38x with int8)", "decode 1.04x"], "teal", 12, title="vs Vulkan")
    s.box(390, 300, 330, 100, ["GenAI vs ggml-openvino:", "prefill 1.27x at pp512", "decode 1.10x (root-caused, step 3)"], "purple", 12, title="vs native")
    s.box(740, 300, 330, 100, ["sync with dev_backend_openvino: +0.4%", "newer OpenVINO: about -1.5% decode", "defaults within 1% of tuned settings"], "gray", 12, title="Other effects")
    s.text(550, 430, "GenAI prefill = prompt tokens / first-token latency at an exact 128/512-token prompt; decode = llm_bench 2nd-token throughput.", 11, MUTED, 400, "middle")
    s.save("d20-final-perf.svg")


def d21_memory_problem():
    s = Svg(1100, 470, "Step 5 - the memory problem: fine while running, too big while loading",
            "Measured system-wide on the iGPU (2026-11-04), before any fix. Peak = during compile_model.")
    rows = [("ggml-openvino, default", 34.5, "red", ""), ("+ SPILL_DIR + MEMORY_OPTIMIZE", 28.2, "red", "ggml's own memory knobs"),
            ("OpenVINO GenAI", 20.9, "orange", ""), ("ggml-vulkan", 7.7, "gray", ""), ("steady decode, ggml-openvino", 6.8, "green", "after load")]
    y = hbars(s, 40, 110, rows, 40, 520, bh=30, gap=14, unit=" GiB", label_w=260, fmt="{:.1f}", ticks=[0, 8, 24, 32, 40])
    x16 = 40 + 260 + 520 * 16 / 40
    s.line(x16, 96, x16, y, C["red"][0], 2, True)
    s.text(x16 + 6, 92, "16 GB machine", 12, C["red"][0], 600)
    s.note(40, 340, 1020, ["The handoff's working theory: 'the GPU plugin needs ~20 GiB = 3x the weights to compile ANY model this size'.",
                           "That would have meant: only a pre-compiled cache can help. It turned out to be wrong - plain compile is 1x the weights,",
                           "and the extra came from three specific, fixable things (next diagrams)."], "orange")
    s.save("d21-memory-problem.svg")


def d22_memory_accounting():
    s = Svg(1100, 520, "Measuring memory on an integrated GPU - four kinds, only two of them are visible to the usual tools",
            "Getting this wrong produced several false conclusions; the fixes were only found after switching to exact per-process numbers.")
    kinds = [("CPU heap (RssAnon)", ["process memory: ggml copies,", "transformed model, compiler", "visible in RSS and cgroups"], "blue", "measure: /proc/pid/status"),
             ("GPU buffers (GTT)", ["weights + activations on the GPU,", "but physically in system RAM", "NOT in RSS, NOT charged to cgroups"], "teal", "measure: xe fdinfo drm-total-gtt"),
             ("page cache (file)", ["mmapped GGUF / spill files", "clean pages, reclaimable", "counts as 'available'"], "gray", "usually ignore"),
             ("TTM page pool", ["GPU pages freed by the process", "kept by the driver for reuse", "not 'available' until shrunk"], "orange", "why 'after compile' looked huge")]
    for i, (t, l, col, m) in enumerate(kinds):
        x = 40 + i * 262
        s.box(x, 90, 242, 120, l, col, 12, title=t)
        s.text(x + 121, 230, m, 11, C[col][0], 600, "middle")
    s.box(40, 260, 500, 110, ["cgroup memory.max=16G 'passes' while the box loses 34 GiB:", "GPU buffers are not memcg-charged.",
                              "System-wide MemAvailable swung by GBs (other users), even", "went negative. Freed GPU memory stays 'used' (TTM pool)."], "red", 12, title="What lied")
    s.box(560, 260, 500, 110, ["peak of (RssAnon + drm-total-gtt) of the process,", "sampled every 100 ms (memprobe/phases.py)",
                               "+ an LD_PRELOAD heap tracker that records live bytes per", "call stack, to name the code that allocates"], "green", 12, title="What worked")
    s.note(40, 392, 1020, ["Faithful '16 GB' test: physically take RAM away. Hugepage reservation failed on this box (27 days uptime -> fragmented,",
                           "stopped at 41 GiB, and MemAvailable overstated free memory under a big pool). An mlock'd 47 GiB 'hog' process worked (step 6)."])
    s.save("d22-memory-accounting.svg")


def d23_cause_embedding():
    s = Svg(1100, 470, "Cause 1 (~9.4 GiB) - the u2 token embedding was decompressed to f16 at compile time",
            "The lookup table is 248 320 x 5120 ternary weights: 303 MiB as u2, 2.37 GiB as f16.")
    s.text(40, 92, "Before: the 'compressed gather' pattern accepted u8/i8/u4/i4 only - not u2", 13, C["red"][0], 600)
    chain = [("u2 table", "303 MiB"), ("Convert f16", "2.37 GiB"), ("Subtract zp", "2.37 GiB"), ("Multiply scale", "2.37 GiB"), ("Gather rows", "per token")]
    for i, (a, b) in enumerate(chain):
        x = 40 + i * 205
        s.box(x, 104, 180, 56, [b], "teal" if i == 0 else ("red" if i < 4 else "gray"), 12, title=a)
        if i:
            s.arrow(x - 25, 132, x, 132)
    s.lines(40, 186, ["The plugin constant-folds the whole chain on its own: several 2.37 GiB intermediates exist at once (~7 GiB transient),",
                      "and the final f16 table stays resident (+2.5 GiB). GenAI never hits this: it keeps embeddings in a separate model."], 12, C["red"][0])
    s.text(40, 250, "After: u2 added to the pattern and to the GPU gather kernel", 13, C["green"][0], 600)
    s.box(40, 262, 380, 64, ["u2 table stays 303 MiB on the GPU"], "teal", 12, title="u2 table")
    s.box(480, 262, 560, 64, ["GatherCompressed: decode only the rows that are looked up,", "(code >> 2*(i%4)) & 3, then (code - zp) x scale"], "green", 12, title="one op")
    s.arrow(420, 294, 480, 294, "green")
    s.box(40, 350, 1000, 88, ["new kernel bit-exact vs a host reference (which is itself bit-exact vs the old unfused path), incl. first and last row",
                              "exec graph: gather_ref__f16 -> gather_ref__u2;  CPU plugin explicitly keeps its old behaviour (no u2 gather there)",
                              "pure-OpenVINO compile of the ggml graph: 17.6 -> 8.4 GiB (TernOCL off), 27.9 -> 18.5 GiB (TernOCL on)"], "green", 12, title="Verified")
    s.save("d23-cause1-embedding.svg")


def d24_cause_constprop():
    s = Svg(1100, 500, "Cause 2 (~5 GiB) - all weight conversions ran in one batch",
            "propagate_constants evaluates constant sub-graphs (e.g. TernOCL's weight re-packing) in one network, then drops the inputs.")

    def panel(x, title, col, chunks):
        s.text(x, 100, title, 13, C[col][0], 600)
        ax, ay, aw, ah = x + 30, 120, 440, 200
        s.line(ax, ay + ah, ax + aw, ay + ah, MUTED)
        s.line(ax, ay, ax, ay + ah, MUTED)
        s.text(ax - 6, ay + 8, "GiB", 11, MUTED, 400, "end")
        s.text(ax + aw, ay + ah + 18, "time ->", 11, MUTED, 400, "end")
        sc = ah / 14
        if not chunks:
            # inputs 6 flat, outputs ramp 0..6 then inputs drop
            s.path(f"M {ax} {ay+ah-6*sc} L {ax+300} {ay+ah-6*sc} L {ax+300} {ay+ah} ", "blue", 2)
            s.path(f"M {ax} {ay+ah-6*sc} L {ax+300} {ay+ah-12*sc} L {ax+440} {ay+ah-6*sc}", "red", 2.5)
            s.text(ax + 300, ay + ah - 12 * sc - 8, "peak ~ inputs 6 + outputs 6", 12, C["red"][0], 600, "middle")
        else:
            pts = []
            n = 8
            for k in range(n + 1):
                t = ax + k * 300 / n
                pts.append((t, 6 + (0.7 if 0 < k < n else 0)))
            d = "M " + " L ".join(f"{t} {ay+ah-v*sc}" for t, v in pts)
            s.path(d + f" L {ax+440} {ay+ah-6*sc}", "green", 2.5)
            s.text(ax + 150, ay + ah - 6.7 * sc - 10, "~ 6 + one chunk", 12, C["green"][0], 600, "middle")
        s.text(ax + 10, ay + ah - 3 * sc, "u2 inputs", 11, C["blue"][0])
    panel(40, "Before: one network over everything", "red", False)
    panel(580, "After: chunks of ~512 MiB, inputs freed per chunk", "green", True)
    s.box(40, 350, 1020, 100, ["split the constant graph into independent pieces (union-find), run one small network per ~512 MiB of inputs,",
                               "then release the inputs whose users are all constants (the program deletes them afterwards anyway)",
                               "GPU-buffer peak 14.0 -> 8.8 GiB = its final level: no transient left. Knob for A/B: OV_GPU_CONST_PROP_CHUNK_MB (0 = old)",
                               "general fix: should help every model whose weights get re-ordered/re-packed at compile time, e.g. oneDNN on dGPUs (not measured)"],
          "green", 12, title="The fix")
    s.save("d24-cause2-constprop.svg")


def d25_cause_fused():
    s = Svg(1100, 520, "Cause 3 (~4.2 GiB and ~48 s) - host copies of merged weights",
            "On the TernOCL path the plugin merges Q/K/V and gate/up into single matmuls; the merged weights were new heap copies.")
    for i, t in enumerate(["W_gate", "W_up"]):
        s.box(40, 100 + i * 70, 150, 50, [t, "(file / ggml memory)"], "gray", 11)
    s.arrow(190, 125, 250, 150)
    s.arrow(190, 195, 250, 170)
    s.box(250, 125, 220, 70, ["concat_and_fold:", "new host copy of both"], "red", 12)
    s.arrow(470, 160, 520, 160)
    s.box(520, 125, 200, 70, ["upload to GPU", "(constant.cpp)"], "teal", 12)
    s.arrow(720, 160, 770, 160)
    s.box(770, 110, 290, 100, ["host copy stays alive for the", "whole program build (~60 s),", "next to its GPU copy and the", "TernOCL-packed copy"], "red", 12)
    s.lines(40, 248, ["Also: folding used get_constant_from_source, which evaluates value bounds on the concatenated u2 data - slow.",
                      "That was most of the '48 s TernOCL compile cost' measured before. Compile: ~63 s -> ~10 s."], 12, C["red"][0])
    s.text(40, 306, "After: an evictable buffer, using a hook the plugin already calls after each upload", 13, C["green"][0], 600)
    steps = [("concat into an", "anonymous mmap"), ("upload to GPU", "as before"), ("hint_evict():", "madvise(DONTNEED)"), ("read again?", "re-fill from sources")]
    for i, (a, b) in enumerate(steps):
        x = 40 + i * 260
        s.box(x, 320, 230, 60, [a, b], "green", 12)
        if i:
            s.arrow(x - 30, 350, x, 350, "green")
    s.note(40, 400, 1020, ["Why madvise and not free(): the plugin de-duplicates uploads by the raw data pointer. A freed address can be reused by the",
                           "next allocation and silently alias another constant's weights. madvise releases the pages but keeps the address reserved.",
                           "The buffer reports an 'invalid source' descriptor so the weight-sharing code ignores it; non-Linux falls back to a plain buffer."])
    s.save("d25-cause3-fused-weights.svg")


def _series(name):
    rows = []
    for l in open(os.path.join(DATA, name)):
        m = re.match(r"\s+ts\s+([\d.]+)\s+sys\s+(-?[\d.]+)\s+anon\s+([\d.]+)\s+gtt\s+([\d.]+)", l)
        if m:
            rows.append(tuple(float(v) for v in m.groups()))
    return rows


def d26_compile_timeline():
    s = Svg(1100, 520, "The compile, measured: before and after the three fixes (same graph, same GPU)",
            "Pure-OpenVINO compile of the ggml-openvino Bonsai 2 graph, TernOCL on. Sampled every second: CPU heap and GPU buffers of the process.")
    for k, (fn, x, title) in enumerate([("series_before.txt", 40, "Before (OpenVINO master)"), ("series_after.txt", 580, "After (3 plugin fixes)")]):
        data = _series(fn)
        ax, ay, aw, ah = x + 40, 110, 440, 270
        tmax, vmax = 72, 30
        s.text(x, 96, title, 14, INK, 600)
        for g in range(0, 31, 10):
            yy = ay + ah - g / vmax * ah
            s.line(ax, yy, ax + aw, yy, GRID)
            s.text(ax - 6, yy + 4, f"{g}", 10, MUTED, 400, "end")
        for t in range(0, 73, 20):
            xx = ax + t / tmax * aw
            s.text(xx, ay + ah + 16, f"{t}s", 10, MUTED, 400, "middle")
        s.line(ax, ay + ah, ax + aw, ay + ah, MUTED)
        y16 = ay + ah - 16 / vmax * ah
        s.line(ax, y16, ax + aw, y16, C["red"][0], 1.5, True)
        s.text(ax + aw, y16 - 5, "16 GiB", 11, C["red"][0], 600, "end")

        def poly(vals, col, sw=2.2):
            d = "M " + " L ".join(f"{ax + t / tmax * aw:.1f} {ay + ah - v / vmax * ah:.1f}" for t, v in vals)
            s.path(d, col, sw)
        poly([(r[0], r[2]) for r in data], "blue")
        poly([(r[0], r[3]) for r in data], "teal")
        poly([(r[0], r[2] + r[3]) for r in data], "red", 2.8)
        pk = max(data, key=lambda r: r[2] + r[3])
        s.text(ax + pk[0] / tmax * aw, ay + ah - (pk[2] + pk[3]) / vmax * ah - 8, f"peak {pk[2] + pk[3]:.1f} GiB", 12, C["red"][0], 600, "middle")
    s.rect(80, 420, 14, 4, C["red"][0])
    s.text(100, 425, "total (heap + GPU)", 12)
    s.rect(260, 420, 14, 4, C["teal"][0])
    s.text(280, 425, "GPU buffers (GTT)", 12)
    s.rect(430, 420, 14, 4, C["blue"][0])
    s.text(450, 425, "CPU heap (RssAnon)", 12)
    s.note(40, 444, 1020, ["Before: the heap grows for ~60 s (merged-weight copies + bound evaluation), then GPU buffers ramp to 23 GiB as everything is uploaded",
                           "and repacked at once. After: merged copies released as they are uploaded, constants processed in chunks - done in ~10 s.",
                           "(1-second samples; the 100 ms-sampled peaks of the same runs are 27.9 and 9.1 GiB.)"])
    s.save("d26-compile-timeline.svg")


def d27_memory_waterfall():
    s = Svg(1100, 450, "Step 5 result - peak memory, fix by fix",
            "Process peak (CPU heap + GPU buffers). Same model, Arc B390 iGPU.")
    s.text(40, 96, "Pure OpenVINO compile of the ggml graph (TernOCL on)", 14, INK, 600)
    hbars(s, 40, 112, [("before", 27.9, "red", ""), ("+ cause 1: u2 GatherCompressed", 18.5, "orange", "-9.4"),
                       ("+ cause 2: chunked const-prop", 13.4, "orange", "-5.1"), ("+ cause 3: evictable merged weights", 9.1, "green", "-4.3")],
          30, 330, label_w=270, fmt="{:.1f}", unit=" GiB")
    s.text(40, 270, "llama-cli end to end, primes prompt, GGML_OPENVINO_MEMORY_OPTIMIZE=1 + SPILL_DIR", 14, INK, 600)
    hbars(s, 40, 286, [("before", 28.2, "red", "~80 s run"), ("after (all three)", 10.0, "green", "30 s run, same output, 13.2 t/s")],
          30, 330, label_w=270, fmt="{:.1f}", unit=" GiB")
    s.box(760, 96, 300, 150, ["GenAI's own IR: 18.0 -> 8.8 GiB", "TernOCL off: 17.6 -> 8.4 GiB", "without ggml's memory knobs:", "33.6 -> 16.5 GiB (ggml keeps", "a 7.3 GiB host weight copy)"], "gray", 12, title="Also")
    s.save("d27-memory-waterfall.svg")


def d28_16gb_test():
    s = Svg(1100, 470, "Step 6 - the faithful 16 GB test (2026-11-09)",
            "Physically take RAM away, run the real workload, watch swap and the OOM killer.")
    ax, w, y, h = 40, 1020, 110, 60
    total = 62.4
    sc = w / total

    def seg(x, v, col, lab):
        s.rect(ax + x * sc, y, v * sc, h, C[col][1], C[col][0])
        s.text(ax + (x + v / 2) * sc, y + 36, lab, 12, C[col][0], 600, "middle")
    seg(0, 47.0, "gray", "47.0 GiB locked by hog.py (mlock) = removed RAM")
    seg(47.0, 15.4, "orange", "15.4 GiB 'laptop'")
    s.text(ax, y - 10, "62.4 GiB machine", 12, MUTED)
    lap_x, lw = 600, 460
    s.text(lap_x, 214, "Inside the 15.4 GiB at the worst moment", 13, INK, 600)
    parts = [(9.2, "red", "our process, peak 9.2"), (6.2, "gray", "everything else (6.2)")]
    xx = lap_x
    for v, col, lab in parts:
        s.rect(xx, 226, v / 15.4 * lw, 40, C[col][1], C[col][0])
        if lab:
            s.text(xx + v / 15.4 * lw / 2, 251, lab, 11, C[col][0], 600, "middle")
        xx += v / 15.4 * lw
    s.text(lap_x, 286, "everything else = ~4.3 GiB of residents (VS Code servers ~ a desktop, other users,", 11, MUTED)
    s.text(lap_x, 300, "kernel), partly pushed to swap at the peak, + reclaimable cache. Min available: 4.2 GiB", 11, MUTED)
    s.box(40, 200, 520, 200, ["output: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29  (exact)", "no OOM kill, safety watchdog never fired",
                              "our peak 9.2 GiB, of which at most 0.8 GiB ever swapped", "~3.4 GiB of OTHER processes swapped out during the",
                              "~15 s weight upload, then swapping nearly stops", "decode 12.8 t/s (13.2 without the limit)"], "green", 12, title="PASS")
    s.box(600, 318, 460, 86, ["hugepage reservation: stopped at 41 GiB (fragmentation)", "and swapped hard at '17 GiB available'",
                              "mlock'd 4 KiB pages need no contiguity -> works"], "orange", 12, title="Method lesson")
    s.text(550, 440, "Without the three fixes the load peak alone (28.2 GiB) is larger than the whole 16 GB machine.", 13, MUTED, 400, "middle")
    s.save("d28-16gb-test.svg")


def d29_where_things_are():
    s = Svg(1100, 470, "Where everything lives, and what is still open",
            "Code state as of 2026-11-09.")
    s.box(40, 90, 330, 180, ["cavusmustafa/llama.cpp", "prism_ov_ternary (step 1)", "prism_ov_ternary_kv4d (step 3)", "prism_ov_ternary_sync (step 4)",
                             "PrismML fork base: PQ2_0 is fork-local", "no PR open"], "blue", 12, title="llama.cpp (pushed)")
    s.box(390, 90, 330, 180, ["upstream master contains PR #38460", "(TernOCL) since 2026-10", "memory fixes: uncommitted patch,", "8 files, built in build-dbgcaps",
                              "memprobe/ov_compile_memory_fixes.patch"], "teal", 12, title="OpenVINO")
    s.box(740, 90, 320, 180, ["memprobe/phases.py (anon+GTT timeline)", "memprobe/ovcompile (pure-OV repro)", "memprobe/heaptrace (alloc stacks)",
                              "memprobe/limit16.sh + hog.py", "bonsai-ov/gpu_busy.py (xe PMU)"], "gray", 12, title="Tools")
    s.box(40, 290, 1020, 150, ["1. memory fixes -> upstream OpenVINO: unit tests, Windows eviction path, measure const-prop chunking on a dGPU",
                               "2. device-resident GDN recurrent state: ~+4.7% decode (-> ~14.0 t/s, gap to native ~1.04x)",
                               "3. PTQ1_0 (Bonsai 1, 1.58-bit) in ggml-openvino - not supported yet",
                               "4. upstreaming the ternary backend work is blocked until PQ2_0 exists outside the PrismML fork",
                               "5. int8 prefill as default needs a real accuracy run (GSM8K-style), not a 90-token diff"], "orange", 12, title="Next steps", align="start")
    s.save("d29-state-and-next.svg")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("d") and name[1:3].isdigit() and callable(fn):
            fn()
    print("\n".join(sorted(os.listdir(OUT))))
