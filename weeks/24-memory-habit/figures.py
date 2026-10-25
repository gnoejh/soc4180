"""Draw the deck's measured figure: memory against habit, by how many situations share the memory.

    uv run python weeks/24-memory-habit/figures.py          # redraw from traces/late.json
    uv run python weeks/24-memory-habit/figures.py --run    # re-measure first (about 2 minutes, in parallel)

--run runs lab_habit.py on ubrain's laptop bandit at each S in SITUATIONS (no robot) and on
the robot's cached world (traces/world.json), and keeps each brain's reward once a
situation has been seen more than 60 times. The slide it writes, slides/fig_late.html, is
a source of the deck like the others.
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
SITUATIONS = (2, 16, 64, 256)
TRIALS = {2: 4000, 16: 4000, 64: 4000, 256: 20000}
BRAINS = ("striatum", "memory", "chooser")
COLOR = {"striatum": "#E07A2F", "memory": "#2F6FB0", "chooser": "#14213D"}
LABEL = {"striatum": "striatum (habit)", "memory": "memory (episodic control)", "chooser": "chooser (the hand-over)"}
SANS = "font-family:'IBM Plex Sans', Arial, sans-serif"
FOOT = "Memory first, habit later"


def run_all() -> None:
    TRACES.mkdir(exist_ok=True)
    jobs = [(f"bandit {S}", ["--bandit", str(S), "--trials", str(TRIALS[S]), "--seeds", "10" if S < 256 else "5"])
            for S in SITUATIONS] + [("robot", [])]
    procs = [subprocess.Popen([sys.executable, str(HERE / "lab_habit.py"), *args, "--name", f"figures {n}"],
                              stdout=subprocess.DEVNULL) for n, args in jobs]
    for p in procs:
        if p.wait():
            raise SystemExit("a run failed")
    rows = list(csv.DictReader(open(HERE / "results.csv", encoding="utf-8")))
    out = {}
    for n, _ in jobs:
        out[n] = {r["brain"]: [float(x) for x in r["curve"].split()]
                  for r in rows if r["name"] == f"figures {n}"}
    (TRACES / "late.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def svg(pts: dict) -> str:
    W, H, L, R, T, B = 1660, 600, 130, 360, 30, 80
    X = lambda S: L + (W - L - R) * SITUATIONS.index(S) / (len(SITUATIONS) - 1)
    lo, hi = 0.5, 0.85
    Y = lambda v: T + (H - T - B) * (hi - v) / (hi - lo)
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="{SANS}">']
    for v in (0.5, 0.6, 0.7, 0.8):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}" stroke="#D9D4C7"/>')
        out.append(f'<text x="{L - 14}" y="{Y(v) + 8}" text-anchor="end" font-size="22" fill="#6B7385">{v:.1f}</text>')
    for S in SITUATIONS:
        out.append(f'<text x="{X(S)}" y="{H - B + 34}" text-anchor="middle" font-size="22" fill="#6B7385">{S}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 14}" text-anchor="middle" font-size="24" fill="#3A4458">situations sharing the memory (laptop bandit, 2 actions paying 0.7 / 0.3)</text>')
    out.append(f'<text x="28" y="{(T + H - B) / 2}" text-anchor="middle" font-size="24" fill="#3A4458" transform="rotate(-90 28 {(T + H - B) / 2})">reward, late (situation seen 61+ times)</text>')
    for v, lab in ((0.70, "always the best 0.70"), (0.58, "matching 0.58")):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}" stroke="#8FA3BF" stroke-dasharray="6 6"/>')
        out.append(f'<text x="{L + 12}" y="{Y(v) - 8}" font-size="20" fill="#6B7385">{lab}</text>')
    for k, b in enumerate(BRAINS):
        line = " ".join(f"{X(S):.1f},{Y(pts[f'bandit {S}'][b][-1]):.1f}" for S in SITUATIONS)
        out.append(f'<polyline points="{line}" fill="none" stroke="{COLOR[b]}" stroke-width="4"/>')
        for S in SITUATIONS:
            out.append(f'<circle cx="{X(S):.1f}" cy="{Y(pts[f"bandit {S}"][b][-1]):.1f}" r="8" fill="#F6F4EE" stroke="{COLOR[b]}" stroke-width="4"/>')
        out.append(f'<line x1="{W - R + 14}" x2="{W - R + 54}" y1="{T + 150 + 36 * k}" y2="{T + 150 + 36 * k}" stroke="{COLOR[b]}" stroke-width="4"/>'
                   f'<text x="{W - R + 62}" y="{T + 158 + 36 * k}" font-size="22" fill="#3A4458">{LABEL[b]}</text>')
    rob = pts["robot"]
    out.append(f'<text x="{W - R + 14}" y="{T + 290}" font-size="22" fill="#14213D">the G1, 2 cues, 3 moves</text>')
    out.append(f'<text x="{W - R + 14}" y="{T + 320}" font-size="22" fill="#3A4458">(always right 0.85):</text>')
    for k, b in enumerate(BRAINS):
        out.append(f'<text x="{W - R + 14}" y="{T + 352 + 30 * k}" font-size="22" fill="{COLOR[b]}">{b} {rob[b][-1]:.2f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args(argv)
    if a.run:
        run_all()
    pts = json.loads((TRACES / "late.json").read_text(encoding="utf-8"))
    order = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))["order"]
    n = order.index("fig_late") + 1
    notes = "; ".join(f"{k}: " + ", ".join(f"{b} {v[b][-1]:.2f}" for b in BRAINS) for k, v in pts.items())
    html = f"""<section id="fig_late" data-transition="fade" style="background:#F6F4EE;color:#14213D;{SANS};padding:110px 128px 140px;display:flex;flex-direction:column;gap:24px">
<div style="display:flex;flex-direction:column;gap:12px">
<p style="font-size:24px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#B0521A">Part 2 · Measured</p>
<h2 style="font-size:56px;font-weight:700;line-height:1.1">When does habit overtake memory?</h2>
</div>
{svg(pts)}
<p style="font-size:26px;line-height:1.4;color:#3A4458">Memory's late reward falls as more situations share its one store (crosstalk); the striatum stays at
matching. With a few situations memory never gives way; at 256, habit wins late. The G1's task has 2.</p>
<p style="position:absolute;left:128px;bottom:56px;width:1664px;font-size:24px;color:#6B7385">{FOOT} · {n}</p>
<aside>lab_habit.py --bandit S (no robot) and the robot's cached world, last bucket (61+ visits). {notes}.</aside>
</section>
"""
    (HERE / "slides" / "fig_late.html").write_text(html, encoding="utf-8", newline="\n")
    print("wrote slides/fig_late.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
