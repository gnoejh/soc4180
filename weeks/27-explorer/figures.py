"""Draw the deck's measured figure: what the walking G1 recognised, glance by glance, without and with a heading hold.

    uv run --extra vision weeks/27-explorer/figures.py          # redraw from traces/walk_*.json
    uv run --extra vision weeks/27-explorer/figures.py --run    # re-measure first (about a minute)

--run walks lab_explore.py twice (--steer 0 and --steer 0.5) and keeps each run's glances
in traces/walk_steer<K>.json. The slide it writes, slides/fig_walk.html, is a source of the
deck like the others.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACES = HERE / "traces"
STEERS = ("0", "0.5")
MIN_STD = 20.0
SANS = "font-family:'IBM Plex Sans', Arial, sans-serif"
FOOT = "The explorer"


def run_all() -> None:
    TRACES.mkdir(exist_ok=True)
    for k in STEERS:
        subprocess.run([sys.executable, str(HERE / "lab_explore.py"), "--steer", k, "--name", f"figure steer {k}"],
                       check=True, stdout=subprocess.DEVNULL)
        shutil.copy(HERE / "last_run.json", TRACES / f"walk_steer{k}.json")


def panel(run: dict, x0: float, title: str) -> list[str]:
    names = run["standing"]
    W, H, L, T, B = 800, 640, 150, 50, 70
    X = lambda s: x0 + L + (W - L - 20) * s / 58.0
    Y = lambda i: T + (H - T - B) * (i + 0.5) / len(names)
    out = [f'<text x="{x0 + L}" y="{T - 18}" font-size="24" fill="#14213D" font-weight="600">{title}</text>']
    seen = set(run["written"])
    for i, n in enumerate(names):
        col = "#14213D" if n in seen else "#B0B6C2"
        out.append(f'<text x="{x0 + L - 10}" y="{Y(i) + 6}" text-anchor="end" font-size="17" fill="{col}">{n}</text>')
        out.append(f'<line x1="{x0 + L}" x2="{x0 + W - 20}" y1="{Y(i)}" y2="{Y(i)}" stroke="#ECE8DD"/>')
    for s in (0, 10, 20, 30, 40, 50):
        out.append(f'<text x="{X(s):.1f}" y="{H - B + 28}" text-anchor="middle" font-size="18" fill="#6B7385">{s}</text>')
    out.append(f'<text x="{x0 + L + (W - L) / 2}" y="{H - 16}" text-anchor="middle" font-size="20" fill="#3A4458">seconds of walking</text>')
    for g in run["glances"]:
        if g["spread"] < MIN_STD:
            continue
        i = names.index(g["guess"])
        if g["stored"]:
            out.append(f'<circle cx="{X(g["tick"] / 50):.1f}" cy="{Y(i):.1f}" r="9" fill="#E07A2F"/>')
        else:
            out.append(f'<circle cx="{X(g["tick"] / 50):.1f}" cy="{Y(i):.1f}" r="3.5" fill="#2F6FB0" opacity="0.6"/>')
    return out


def svg(runs: dict) -> str:
    W, H = 1660, 640
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="{SANS}">']
    for j, k in enumerate(STEERS):
        r = runs[k]
        title = (f"no heading hold: {len(r['written'])} written" if k == "0"
                 else f"heading hold (steer {k}): {len(r['written'])} written")
        out += panel(r, 830 * j, title)
    out.append("</svg>")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args(argv)
    if a.run:
        run_all()
    runs = {k: json.loads((TRACES / f"walk_steer{k}.json").read_text(encoding="utf-8")) for k in STEERS}
    order = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))["order"]
    n = order.index("fig_walk") + 1
    notes = "; ".join(f"steer {k}: written {' '.join(r['written'])}" for k, r in runs.items())
    html = f"""<section id="fig_walk" data-transition="fade" style="background:#F6F4EE;color:#14213D;{SANS};padding:100px 128px 140px;display:flex;flex-direction:column;gap:18px">
<div style="display:flex;flex-direction:column;gap:12px">
<p style="font-size:24px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#B0521A">Part 2 · Measured</p>
<h2 style="font-size:52px;font-weight:700;line-height:1.1">What the eye recognised, glance by glance</h2>
</div>
{svg(runs)}
<p style="font-size:24px;line-height:1.4;color:#3A4458">Small dots: a glance recognised as that picture. Large: written into the episode (the third steady glance).
Rows in the gallery's order, grey where never written. Without a heading hold the robot drifts off the line after half a minute and writes nothing new.</p>
<p style="position:absolute;left:128px;bottom:56px;width:1664px;font-size:24px;color:#6B7385">{FOOT} · {n}</p>
<aside>lab_explore.py --steer 0 and --steer 0.5, 58 s at 0.5 m/s. {notes}.</aside>
</section>
"""
    (HERE / "slides" / "fig_walk.html").write_text(html, encoding="utf-8", newline="\n")
    print("wrote slides/fig_walk.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
