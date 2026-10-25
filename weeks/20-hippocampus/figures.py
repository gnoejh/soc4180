"""Draw the deck's measured figure: the certificate against the robot's memory.

    uv run python weeks/20-hippocampus/figures.py          # redraw from traces/points.json
    uv run python weeks/20-hippocampus/figures.py --run    # re-measure first (about 3 minutes)

--run runs lab_remember.py with 10 episodes at each (D, pushes) in GRID and keeps the
memory rates in traces/points.json. The slide it writes, slides/fig_capacity.html, is a
source of the deck like the others; build.py assembles.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRACES = HERE / "traces"
GRID = [(d, k) for d in (64, 32) for k in (8, 16, 24)]
KS = tuple(range(2, 25, 2))
COLOR = {64: "#2F6FB0", 32: "#E07A2F"}
SANS = "font-family:'IBM Plex Sans', Arial, sans-serif"
FOOT = "One-shot memory"


def run_all() -> None:
    TRACES.mkdir(exist_ok=True)
    out = {}
    for d, k in GRID:
        name = f"figures d{d} k{k}"
        print(f"measuring D {d}, {k} pushes ...", flush=True)
        subprocess.run([sys.executable, str(HERE / "lab_remember.py"), "--episodes", "10", "--d", str(d),
                        "--pushes", str(k), "--name", name], check=True, stdout=subprocess.DEVNULL)
        rows = list(csv.DictReader(open(HERE / "results.csv", encoding="utf-8")))
        r = next(x for x in reversed(rows) if x["name"] == name)
        out[f"{d},{k}"] = {"memory": float(r["memory"]), "perception": float(r["perception"]),
                          "certificate": float(r["certificate"])}
    (TRACES / "points.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def curves() -> dict[int, list[float]]:
    sys.path.insert(0, str(HERE))
    import ubrain_local
    ubrain_local.load()
    from ubrain.codecs import GaussianCodes, certify
    return {d: list(certify(GaussianCodes(dim=d), np.arange(8), d=d, ks=KS, n_trials=3000).accuracy) for d in (64, 32)}


def svg(cur: dict, pts: dict) -> str:
    W, H, L, R, T, B = 1660, 600, 130, 260, 30, 80
    X = lambda k: L + (W - L - R) * (k - 2) / 22
    Y = lambda v: T + (H - T - B) * (1.0 - v)
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="{SANS}">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}" stroke="#D9D4C7"/>')
        out.append(f'<text x="{L - 14}" y="{Y(v) + 8}" text-anchor="end" font-size="22" fill="#6B7385">{v:.2f}</text>')
    for k in (4, 8, 12, 16, 20, 24):
        out.append(f'<text x="{X(k)}" y="{H - B + 34}" text-anchor="middle" font-size="22" fill="#6B7385">{k}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 14}" text-anchor="middle" font-size="24" fill="#3A4458">pushes in the episode</text>')
    out.append(f'<text x="28" y="{(T + H - B) / 2}" text-anchor="middle" font-size="24" fill="#3A4458" transform="rotate(-90 28 {(T + H - B) / 2})">recalled as perceived</text>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(1 / 8)}" y2="{Y(1 / 8)}" stroke="#8FA3BF" stroke-dasharray="6 6"/>')
    out.append(f'<text x="{W - R + 16}" y="{Y(1 / 8) + 8}" font-size="22" fill="#6B7385">chance 0.125</text>')
    for d in (64, 32):
        line = " ".join(f"{X(k):.1f},{Y(a):.1f}" for k, a in zip(KS, cur[d]))
        out.append(f'<polyline points="{line}" fill="none" stroke="{COLOR[d]}" stroke-width="4"/>')
        out.append(f'<text x="{W - R + 16}" y="{Y(cur[d][-1]) + 8:.1f}" font-size="24" fill="{COLOR[d]}">D = {d}</text>')
        for key, p in pts.items():
            dd, k = (int(x) for x in key.split(","))
            if dd == d:
                out.append(f'<circle cx="{X(k):.1f}" cy="{Y(p["memory"]):.1f}" r="11" fill="#F6F4EE" stroke="{COLOR[d]}" stroke-width="5"/>')
    ly = Y(0.36)                                      # the legend, in the empty lower left
    out.append(f'<line x1="{L + 20}" x2="{L + 70}" y1="{ly}" y2="{ly}" stroke="#14213D" stroke-width="4"/>'
               f'<text x="{L + 80}" y="{ly + 8}" font-size="22" fill="#3A4458">certificate (predicted before anything is stored)</text>'
               f'<circle cx="{L + 45}" cy="{ly + 38}" r="10" fill="#F6F4EE" stroke="#14213D" stroke-width="4"/>'
               f'<text x="{L + 80}" y="{ly + 46}" font-size="22" fill="#3A4458">the G1, 10 episodes</text>')
    out.append("</svg>")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true", help="re-measure first")
    a = ap.parse_args(argv)
    if a.run:
        run_all()
    pts = json.loads((TRACES / "points.json").read_text(encoding="utf-8"))
    order = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))["order"]
    n = order.index("fig_capacity") + 1
    html = f"""<section id="fig_capacity" data-transition="fade" style="background:#F6F4EE;color:#14213D;{SANS};padding:110px 128px 140px;display:flex;flex-direction:column;gap:24px">
<div style="display:flex;flex-direction:column;gap:12px">
<p style="font-size:24px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#B0521A">Part 3 · Measured</p>
<h2 style="font-size:56px;font-weight:700;line-height:1.1">The certificate, and the robot's memory</h2>
</div>
{svg(curves(), pts)}
<p style="font-size:26px;line-height:1.4;color:#3A4458">Lines: the certificate, computed from the codes before a single push. Circles: the G1's
hippocampus, 10 episodes each. A bigger workspace holds more; every episode fades gradually, not at a cliff.</p>
<p style="position:absolute;left:128px;bottom:56px;width:1664px;font-size:24px;color:#6B7385">{FOOT} · {n}</p>
<aside>lab_remember.py --episodes 10 --d D --pushes K, memory = recalled equals perceived. """ + \
        "; ".join(f"D {k.split(',')[0]} K {k.split(',')[1]}: {v['memory']:.2f} vs {v['certificate']:.2f}" for k, v in pts.items()) + \
        """. The certificate's capacity grows in proportion to D.</aside>
</section>
"""
    (HERE / "slides" / "fig_capacity.html").write_text(html, encoding="utf-8", newline="\n")
    print("wrote slides/fig_capacity.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
