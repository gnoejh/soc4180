"""Draw the deck's measured figure: the cortex against its law, in the robot's world and the law's.

    uv run python weeks/22-consolidation/figures.py          # redraw from traces/points.json
    uv run python weeks/22-consolidation/figures.py --run    # re-measure first (about 2 minutes, in parallel)

--run runs lab_consolidate.py at each number of episodes in EPISODES, on the robot and with
--distinct, and keeps the cortex's faithful recall and its law in traces/points.json. The
slide it writes, slides/fig_cortex.html, is a source of the deck like the others.
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
EPISODES = (5, 10, 20, 40)          # x 8 pushes = 40 ... 320 pairs
COLOR = {"robot": "#E07A2F", "distinct": "#2F6FB0"}
LABEL = {"robot": "the robot: 8 kinds of push, repeating", "distinct": "--distinct: every item different"}
SANS = "font-family:'IBM Plex Sans', Arial, sans-serif"
FOOT = "From hippocampus to cortex"


def run_all() -> None:
    TRACES.mkdir(exist_ok=True)
    jobs = [(w, e) for w in ("robot", "distinct") for e in EPISODES]
    procs = [subprocess.Popen([sys.executable, str(HERE / "lab_consolidate.py"), "--episodes", str(e),
                               "--name", f"figures {w} {e}"] + (["--distinct"] if w == "distinct" else []),
                              stdout=subprocess.DEVNULL) for w, e in jobs]
    for p in procs:
        if p.wait():
            raise SystemExit("a run failed")
    rows = list(csv.DictReader(open(HERE / "results.csv", encoding="utf-8")))
    out = {}
    for w, e in jobs:
        r = next(x for x in reversed(rows) if x["name"] == f"figures {w} {e}")
        out.setdefault(w, []).append({"pairs": 8 * e, "cortex": float(r["cortex"]), "law": float(r["law_cortex"])})
    (TRACES / "points.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def svg(pts: dict) -> str:
    W, H, L, R, T, B = 1660, 600, 130, 40, 30, 80
    xs = [40, 80, 160, 320]
    X = lambda n: L + (W - L - R) * xs.index(n) / (len(xs) - 1)
    Y = lambda v: T + (H - T - B) * (1.0 - v)
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="{SANS}">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}" stroke="#D9D4C7"/>')
        out.append(f'<text x="{L - 14}" y="{Y(v) + 8}" text-anchor="end" font-size="22" fill="#6B7385">{v:.2f}</text>')
    for n in xs:
        out.append(f'<text x="{X(n)}" y="{H - B + 34}" text-anchor="middle" font-size="22" fill="#6B7385">{n}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 14}" text-anchor="middle" font-size="24" fill="#3A4458">pushes consolidated into the cortex (episodes of 8)</text>')
    out.append(f'<text x="28" y="{(T + H - B) / 2}" text-anchor="middle" font-size="24" fill="#3A4458" transform="rotate(-90 28 {(T + H - B) / 2})">cortex: recollected as given</text>')
    for k, (w, rows) in enumerate(pts.items()):
        law = " ".join(f"{X(r['pairs']):.1f},{Y(r['law']):.1f}" for r in rows)
        meas = " ".join(f"{X(r['pairs']):.1f},{Y(r['cortex']):.1f}" for r in rows)
        out.append(f'<polyline points="{law}" fill="none" stroke="{COLOR[w]}" stroke-width="3" stroke-dasharray="10 8"/>')
        out.append(f'<polyline points="{meas}" fill="none" stroke="{COLOR[w]}" stroke-width="4"/>')
        for r in rows:
            out.append(f'<circle cx="{X(r["pairs"]):.1f}" cy="{Y(r["cortex"]):.1f}" r="10" fill="#F6F4EE" stroke="{COLOR[w]}" stroke-width="5"/>')
        y = Y(0.30) + 40 * k
        out.append(f'<line x1="{L + 20}" x2="{L + 70}" y1="{y}" y2="{y}" stroke="{COLOR[w]}" stroke-width="4"/>'
                   f'<text x="{L + 82}" y="{y + 8}" font-size="22" fill="#3A4458">{LABEL[w]}</text>')
    out.append(f'<line x1="{L + 20}" x2="{L + 70}" y1="{Y(0.30) + 80}" y2="{Y(0.30) + 80}" stroke="#14213D" stroke-width="3" stroke-dasharray="10 8"/>'
               f'<text x="{L + 82}" y="{Y(0.30) + 88}" font-size="22" fill="#3A4458">dashed: each world\'s law (it counts different pairs)</text>')
    out.append("</svg>")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args(argv)
    if a.run:
        run_all()
    pts = json.loads((TRACES / "points.json").read_text(encoding="utf-8"))
    order = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))["order"]
    n = order.index("fig_cortex") + 1
    notes = "; ".join(f"{w} {r['pairs']}: {r['cortex']:.2f} (law {r['law']:.2f})" for w, rows in pts.items() for r in rows)
    html = f"""<section id="fig_cortex" data-transition="fade" style="background:#F6F4EE;color:#14213D;{SANS};padding:110px 128px 140px;display:flex;flex-direction:column;gap:24px">
<div style="display:flex;flex-direction:column;gap:12px">
<p style="font-size:24px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#B0521A">Part 3 · Measured</p>
<h2 style="font-size:56px;font-weight:700;line-height:1.1">The cortex keeps its law, in the law's world</h2>
</div>
{svg(pts)}
<p style="font-size:26px;line-height:1.4;color:#3A4458">With every item different the cortex follows its law. On the robot, where 8 kinds of push repeat,
it falls far below it: copies of one item add their crosstalk together.</p>
<p style="position:absolute;left:128px;bottom:56px;width:1664px;font-size:24px;color:#6B7385">{FOOT} · {n}</p>
<aside>lab_consolidate.py --episodes E (robot) and --distinct --episodes E, d 64. {notes}.</aside>
</section>
"""
    (HERE / "slides" / "fig_cortex.html").write_text(html, encoding="utf-8", newline="\n")
    print("wrote slides/fig_cortex.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
