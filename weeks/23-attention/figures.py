"""Draw the deck's measured figure: what the gate writes, and what the cortex keeps, by window.

    uv run python weeks/23-attention/figures.py          # redraw from traces/windows.json
    uv run python weeks/23-attention/figures.py --run    # re-measure first (about a minute, in parallel)

--run runs lab_notice.py at each gate window in WINDOWS (random world, defaults otherwise)
and keeps the written share, the consolidated share and the cortex's recall in
traces/windows.json. The slide it writes, slides/fig_window.html, is a source of the deck.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACES = HERE / "traces"
WINDOWS = (2, 4, 8, 16)
SANS = "font-family:'IBM Plex Sans', Arial, sans-serif"
FOOT = "Noticing"
BARS = (("written", "written into memory (share of all pushes)", "#E07A2F"),
        ("consolidated", "of those, replayed into the cortex", "#8FA3BF"),
        ("cortex", "of those, recalled by the cortex", "#2F6FB0"))


def run_all() -> None:
    TRACES.mkdir(exist_ok=True)
    procs = [subprocess.Popen([sys.executable, str(HERE / "lab_notice.py"), "--window", str(w),
                               "--name", f"figures w{w}"], stdout=subprocess.DEVNULL) for w in WINDOWS]
    for p in procs:
        if p.wait():
            raise SystemExit("a run failed")
    rows = list(csv.DictReader(open(HERE / "results.csv", encoding="utf-8")))
    out = {}
    for w in WINDOWS:
        r = next(x for x in reversed(rows) if x["name"] == f"figures w{w}")
        out[str(w)] = {k: float(r[k]) for k in ("written", "consolidated", "cortex", "executive")}
    (TRACES / "windows.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def svg(pts: dict) -> str:
    W, H, L, R, T, B = 1660, 600, 130, 40, 30, 110
    Y = lambda v: T + (H - T - B) * (1.0 - v)
    gw = (W - L - R) / len(pts)
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="{SANS}">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}" stroke="#D9D4C7"/>')
        out.append(f'<text x="{L - 14}" y="{Y(v) + 8}" text-anchor="end" font-size="22" fill="#6B7385">{v:.2f}</text>')
    for k, (w, r) in enumerate(pts.items()):
        for j, (key, _label, color) in enumerate(BARS):
            x = L + k * gw + gw * (0.12 + 0.26 * j)
            v = r[key]
            out.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{gw * 0.24:.1f}" height="{Y(0) - Y(v):.1f}" fill="{color}"/>')
            out.append(f'<text x="{x + gw * 0.12:.1f}" y="{Y(v) - 10:.1f}" text-anchor="middle" font-size="22" fill="#14213D">{v:.2f}</text>')
        out.append(f'<text x="{L + k * gw + gw / 2:.1f}" y="{H - B + 36}" text-anchor="middle" font-size="26" fill="#14213D">window {w}</text>')
    for j, (_key, label, color) in enumerate(BARS):
        x = L + 10 + 500 * j
        out.append(f'<rect x="{x}" y="{H - 40}" width="24" height="18" fill="{color}"/><text x="{x + 34}" y="{H - 24}" font-size="22" fill="#3A4458">{label}</text>')
    out.append("</svg>")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args(argv)
    if a.run:
        run_all()
    pts = json.loads((TRACES / "windows.json").read_text(encoding="utf-8"))
    order = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))["order"]
    n = order.index("fig_window") + 1
    notes = "; ".join(f"window {w}: written {r['written']:.2f}, consolidated {r['consolidated']:.2f}, cortex {r['cortex']:.2f}, executive {r['executive']:.2f}" for w, r in pts.items())
    html = f"""<section id="fig_window" data-transition="fade" style="background:#F6F4EE;color:#14213D;{SANS};padding:110px 128px 140px;display:flex;flex-direction:column;gap:24px">
<div style="display:flex;flex-direction:column;gap:12px">
<p style="font-size:24px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#B0521A">Part 2 · Measured</p>
<h2 style="font-size:56px;font-weight:700;line-height:1.1">The gate trades load against coverage</h2>
</div>
{svg(pts)}
<p style="font-size:26px;line-height:1.4;color:#3A4458">A short window forgets what it has felt, writes most pushes, keeps the brain busy and the cortex crowded.
A long window writes only what is really new, and the cortex keeps nearly all of it.</p>
<p style="position:absolute;left:128px;bottom:56px;width:1664px;font-size:24px;color:#6B7385">{FOOT} · {n}</p>
<aside>lab_notice.py --window W (20 bursts of 6, 8 quiet moments, random world). {notes}.</aside>
</section>
"""
    (HERE / "slides" / "fig_window.html").write_text(html, encoding="utf-8", newline="\n")
    print("wrote slides/fig_window.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
