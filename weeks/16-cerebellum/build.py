"""Build slides.html: the cerebellum deck as one offline page (arrow keys, F for full screen).

    uv run python weeks/16-cerebellum/build.py

The sources are deck.json and slides/<id>.html, copied from
W:/papercode/teaching/lecture_robot/cerebellum_g1, the same files as the claude.ai deck
(https://claude.ai/artifact/Au3WMA4zriFuM96uZ9Xpn2). This script only assembles.
Speaker notes (<aside>) are kept but hidden; N shows them. It also writes
slides_students.html, the same deck without notes. Both outputs are build artifacts.
<x-connector> lines are drawn as inline SVG, since a plain browser does not know them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

CONNECTOR = re.compile(r"<x-connector\s+([^>]*)></x-connector>")
ATTR = re.compile(r'([\w-]+)="([^"]*)"')


def connector_svg(m: re.Match) -> str:
    a = dict(ATTR.findall(m.group(1)))
    x1, y1, x2, y2 = (float(a[k]) for k in ("x1", "y1", "x2", "y2"))
    style = a.get("style", "")
    color = (re.search(r"color:\s*(#[0-9A-Fa-f]{3,8})", style) or [None, "#14213D"])[1]
    width = float((re.search(r"border-width:\s*(\d+)", style) or [None, "2"])[1])
    head = 4 * width
    dx, dy = x2 - x1, y2 - y1
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / n, dy / n
    bx, by = x2 - ux * head, y2 - uy * head            # base of the arrowhead
    px, py = -uy * head * 0.6, ux * head * 0.6
    return (f'<svg class="conn" width="1920" height="1080" viewBox="0 0 1920 1080">'
            f'<line x1="{x1}" y1="{y1}" x2="{bx}" y2="{by}" stroke="{color}" stroke-width="{width}"/>'
            f'<polygon points="{x2},{y2} {bx + px},{by + py} {bx - px},{by - py}" fill="{color}"/></svg>')


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{fonts}
<style>
html, body {{ margin: 0; height: 100%; background: #0B1222; overflow: hidden; }}
#stage {{ position: absolute; left: 50%; top: 50%; width: 1920px; height: 1080px; transform-origin: 0 0; }}
section {{ position: absolute; inset: 0; width: 1920px; height: 1080px; box-sizing: border-box; }}
section:not(.on) {{ display: none !important; }}
section * {{ margin: 0; box-sizing: border-box; }}
section ul, section ol {{ padding-left: 1.2em; }}
section aside {{ display: none; }}
body.notes section.on aside {{ display: block; position: absolute; left: 0; right: 0; bottom: 0;
  background: rgba(11,18,34,.92); color: #F6F4EE; font: 26px/1.4 Arial, sans-serif; padding: 24px 48px; }}
svg.conn {{ position: absolute; left: 0; top: 0; pointer-events: none; }}
#n {{ position: fixed; right: 12px; bottom: 8px; color: #8FA3BF; font: 14px Arial, sans-serif; }}
</style></head>
<body><div id="stage">
{slides}
</div><div id="n"></div>
<script>
const s = [...document.querySelectorAll('section')]; let i = 0;
const stage = document.getElementById('stage');
function fit() {{ const k = Math.min(innerWidth / 1920, innerHeight / 1080);
  stage.style.transform = `scale(${{k}}) translate(-50%, -50%)`; }}
function show(j) {{ i = Math.max(0, Math.min(s.length - 1, j));
  s.forEach((x, k) => x.classList.toggle('on', k === i));
  document.getElementById('n').textContent = `${{i + 1}} / ${{s.length}}`; location.hash = i + 1; }}
addEventListener('resize', fit);
addEventListener('keydown', e => {{
  if (['ArrowRight', 'PageDown', ' '].includes(e.key)) show(i + 1);
  else if (['ArrowLeft', 'PageUp'].includes(e.key)) show(i - 1);
  else if (e.key === 'Home') show(0); else if (e.key === 'End') show(s.length - 1);
  else if (e.key === 'n' || e.key === 'N') document.body.classList.toggle('notes');
  else if (e.key === 'f' || e.key === 'F') document.documentElement.requestFullscreen?.(); }});
addEventListener('click', e => show(e.clientX > innerWidth / 3 ? i + 1 : i - 1));
fit(); show((parseInt(location.hash.slice(1)) || 1) - 1);
</script></body></html>
"""


ASIDE = re.compile(r"<aside>.*?</aside>", re.S)
#: The students' copy: the same slides without speaker notes.
STUDENTS = HERE / "slides_students.html"


def build(out: Path, notes: bool) -> None:
    deck = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))
    fonts = "\n".join(f'<link rel="stylesheet" href="{f["href"]}">'
                      for f in deck["faces"].values() if "href" in f)
    parts = []
    for sid in deck["order"]:
        html = (HERE / "slides" / f"{sid}.html").read_text(encoding="utf-8").strip()
        if not notes:
            html = ASIDE.sub("", html)
        parts.append(CONNECTOR.sub(connector_svg, html))
    out.write_text(PAGE.format(title=deck["title"], fonts=fonts, slides="\n".join(parts)),
                   encoding="utf-8", newline="\n")
    print(f"wrote {out}: {len(parts)} slides, {'with' if notes else 'without'} notes")


def main() -> None:
    build(HERE / "slides.html", notes=True)
    build(STUDENTS, notes=False)


if __name__ == "__main__":
    main()
