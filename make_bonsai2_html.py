#!/usr/bin/env python3
"""Build a portable, standalone HTML version of BONSAI2_EXPLAINED.md.

Single file, no network and no local assets at run time: every referenced SVG is
inlined. The document is long, so each top-level `## ` section becomes a tab.

    python3 make_bonsai2_html.py          # -> BONSAI2_EXPLAINED.html
"""

import html
import pathlib
import re

import markdown

HERE = pathlib.Path(__file__).parent
SRC = HERE / "BONSAI2_EXPLAINED.md"
OUT = HERE / "BONSAI2_EXPLAINED.html"

text = SRC.read_text(encoding="utf-8")

# ---------------------------------------------------------------- split into tabs
# Keep everything before the first `## ` as the intro, then one tab per `## `.
parts = re.split(r"^## (?!#)", text, flags=re.MULTILINE)
intro_md = parts[0]
sections = []
for chunk in parts[1:]:
    title, _, body = chunk.partition("\n")
    sections.append((title.strip(), body))

# The intro carries the H1 + standfirst; show it above the tab strip.
title_match = re.search(r"^# (.+)$", intro_md, flags=re.MULTILINE)
doc_title = title_match.group(1).strip() if title_match else "Bonsai 2 on OpenVINO"
intro_md = re.sub(r"^# .+$", "", intro_md, count=1, flags=re.MULTILINE).strip()
intro_md = intro_md.rstrip("-\n ").strip()

MD_EXT = ["tables", "fenced_code", "sane_lists"]


def render(md_text):
    return markdown.markdown(md_text, extensions=MD_EXT)


def inline_svgs(html_text):
    """Replace <img src="diagrams_bonsai2/x.svg"> with the SVG itself."""
    def sub(m):
        src = m.group("src")
        path = HERE / src
        if not path.exists():
            return m.group(0)
        svg = path.read_text(encoding="utf-8")
        svg = re.sub(r'^<\?xml[^>]*\?>\s*', "", svg)
        # Let the SVG scale to the column instead of its intrinsic 1000px.
        svg = svg.replace("<svg ", '<svg class="diagram" preserveAspectRatio="xMidYMid meet" ', 1)
        svg = re.sub(r'(<svg[^>]*?)\swidth="\d+"\s+height="\d+"', r"\1", svg, count=1)
        return f'<figure class="fig">{svg}</figure>'

    return re.sub(r'<img[^>]*src="(?P<src>[^"]+\.svg)"[^>]*/?>', sub, html_text)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "s"


tabs, panels = [], []
for i, (title, body) in enumerate(sections):
    sid = f"tab-{i}-{slug(title)}"
    # Short label for the strip: drop a trailing clause after " - "
    label = title.split(" - ")[0]
    tabs.append(
        f'<button class="tab" role="tab" id="btn-{sid}" aria-controls="{sid}" '
        f'aria-selected="{"true" if i == 0 else "false"}" tabindex="{0 if i == 0 else -1}">'
        f"{html.escape(label)}</button>"
    )
    panels.append(
        f'<section class="panel{" is-active" if i == 0 else ""}" role="tabpanel" id="{sid}" '
        f'aria-labelledby="btn-{sid}"{"" if i == 0 else " hidden"}>'
        f"<h2>{html.escape(title)}</h2>{inline_svgs(render(body))}</section>"
    )

CSS = """
:root{
  --ink:#1b1f24; --muted:#6b7480; --rule:#dfe4ea; --surface:#ffffff; --panel:#f7f9fb;
  --accent:#2f6fb3; --code-bg:#f4f6f9; --warn-bg:#fdf6e7; --warn-bd:#c77c1e;
}
@media (prefers-color-scheme: dark){
  :root{
    --ink:#e6e9ee; --muted:#9aa4b2; --rule:#2b3240; --surface:#14171c; --panel:#1b1f27;
    --accent:#6ea8e8; --code-bg:#1d2229; --warn-bg:#2a2317; --warn-bd:#c77c1e;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
  font:16px/1.65 "Segoe UI",Helvetica,Arial,sans-serif;-webkit-text-size-adjust:100%}
.wrap{max-width:1060px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:28px;line-height:1.25;margin:0 0 10px}
h2{font-size:22px;margin:0 0 18px;padding-bottom:10px;border-bottom:1px solid var(--rule)}
h3{font-size:17px;margin:30px 0 10px}
p,li{color:var(--ink)}
.standfirst{color:var(--muted);font-size:15px;margin:0 0 20px}
.standfirst p{color:var(--muted);margin:.5em 0}
a{color:var(--accent)}
hr{border:0;border-top:1px solid var(--rule);margin:28px 0}
code{font:13.5px/1.5 ui-monospace,Consolas,monospace;background:var(--code-bg);
  padding:.12em .36em;border-radius:4px}
pre{background:var(--code-bg);border:1px solid var(--rule);border-radius:8px;
  padding:14px 16px;overflow:auto}
pre code{background:none;padding:0}
blockquote{margin:20px 0;padding:12px 18px;background:var(--warn-bg);
  border-left:4px solid var(--warn-bd);border-radius:0 6px 6px 0}
blockquote p{margin:.4em 0}
table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14.5px;display:block;overflow-x:auto}
th,td{border:1px solid var(--rule);padding:8px 11px;text-align:left;vertical-align:top}
th{background:var(--panel);font-weight:600}
tbody tr:nth-child(even){background:color-mix(in srgb,var(--panel) 55%,transparent)}
.fig{margin:22px 0;padding:0;background:#fff;border:1px solid var(--rule);
  border-radius:8px;overflow:hidden}
svg.diagram{display:block;width:100%;height:auto}
.tabs{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:6px;
  padding:10px 0;margin:0 0 24px;background:var(--surface);border-bottom:1px solid var(--rule)}
.tab{font:600 13px/1 "Segoe UI",Helvetica,Arial,sans-serif;color:var(--muted);
  background:transparent;border:1px solid transparent;border-radius:999px;
  padding:8px 13px;cursor:pointer}
.tab:hover{color:var(--ink);background:var(--panel)}
.tab[aria-selected="true"]{color:#fff;background:var(--accent);border-color:var(--accent)}
.tab:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.panel{display:none;animation:fade .18s ease-out}
.panel.is-active{display:block}
@keyframes fade{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:none}}
.foot{margin-top:40px;padding-top:16px;border-top:1px solid var(--rule);
  color:var(--muted);font-size:13px}
@media print{
  .tabs{display:none} .panel{display:block!important;page-break-before:always}
  .panel:first-of-type{page-break-before:auto} body{background:#fff;color:#000}
}
"""

JS = """
(function(){
  var strip=document.querySelector('.tabs');
  var tabs=[].slice.call(document.querySelectorAll('.tab'));
  var panels=[].slice.call(document.querySelectorAll('.panel'));
  function show(i,focus){
    tabs.forEach(function(t,j){
      var on=i===j;
      t.setAttribute('aria-selected',on?'true':'false');
      t.tabIndex=on?0:-1;
      panels[j].classList.toggle('is-active',on);
      panels[j].hidden=!on;
    });
    if(focus) tabs[i].focus();
    if(history.replaceState) history.replaceState(null,'','#'+panels[i].id);
  }
  strip.addEventListener('click',function(e){
    var b=e.target.closest('.tab'); if(b) show(tabs.indexOf(b),false);
  });
  strip.addEventListener('keydown',function(e){
    var i=tabs.indexOf(document.activeElement); if(i<0) return;
    var n={ArrowRight:i+1,ArrowLeft:i-1,Home:0,End:tabs.length-1}[e.key];
    if(n===undefined) return;
    e.preventDefault(); show((n+tabs.length)%tabs.length,true);
  });
  var h=location.hash.slice(1);
  var k=panels.findIndex(function(p){return p.id===h;});
  show(k>=0?k:0,false);
})();
"""

doc = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(doc_title)}</title>
<style>{CSS}</style>
<body>
<div class="wrap">
  <h1>{html.escape(doc_title)}</h1>
  <div class="standfirst">{render(intro_md)}</div>
  <div class="tabs" role="tablist" aria-label="Sections">{''.join(tabs)}</div>
  {''.join(panels)}
  <p class="foot">Standalone build of <code>BONSAI2_EXPLAINED.md</code> — all diagrams inlined,
  no external assets. Regenerate with <code>python3 make_bonsai2_html.py</code>.</p>
</div>
<script>{JS}</script>
</body>
</html>
"""

OUT.write_text(doc, encoding="utf-8")
print(f"wrote {OUT.name}  {len(doc)/1024:.0f} KB  ({len(sections)} tabs)")
