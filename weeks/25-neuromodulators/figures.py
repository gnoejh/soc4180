"""Draw the deck's measured figure: serotonin's patience against the law, for two sure rewards.

    uv run python weeks/25-neuromodulators/figures.py          # redraw from traces/patience.json
    uv run python weeks/25-neuromodulators/figures.py --run    # re-measure first (seconds)

--run runs lab_patience.py serotonin at each fixed tau in TAUS and sure reward in SMALL,
and keeps P(walk on) and the reward per moment in traces/patience.json. The slide it
writes, slides/fig_patience.html, is a source of the deck like the others.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACES = HERE / "traces"
TAUS = (1.5, 2, 3, 4, 6, 10, 20)
SMALL = (0.4, 0.1)
COLOR = {0.4: "#E07A2F", 0.1: "#2F6FB0"}
SANS = "font-family:'IBM Plex Sans', Arial, sans-serif"
FOOT = "One number each"
D_S, D_L, P = 2, 8, 0.55


def p_walk(tau, r_s):
    g = math.exp(-1.0 / tau)
    return P * g ** D_L / (r_s * g ** D_S + P * g ** D_L)


def run_all() -> None:
    TRACES.mkdir(exist_ok=True)
    for r in SMALL:
        for tau in TAUS:
            subprocess.run([sys.executable, str(HERE / "lab_patience.py"), "serotonin", "--tau", str(tau),
                            "--small", str(r)], check=True, stdout=subprocess.DEVNULL)
    rows = list(csv.DictReader(open(HERE / "results.csv", encoding="utf-8")))
    out = {str(r): [] for r in SMALL}
    for row in rows:
        if row["step"] == "serotonin" and row["arm"].endswith("fixed"):
            r = row["extra"].split(";")[0].split()[1]
            if r in out:
                tau = float(row["arm"].split()[1])
                rate = float(row["extra"].split("reward per moment")[1])
                out[r] = [x for x in out[r] if x["tau"] != tau] + [{"tau": tau, "p_walk": float(row["measured"]), "rate": rate}]
    for r in out:
        out[r].sort(key=lambda x: x["tau"])
    (TRACES / "patience.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def svg(pts: dict) -> str:
    W, H, L, R, T, B = 1660, 600, 130, 420, 30, 80
    lt = lambda tau: math.log(tau)
    X = lambda tau: L + (W - L - R) * (lt(tau) - lt(1.5)) / (lt(20) - lt(1.5))
    Y = lambda v: T + (H - T - B) * (1.0 - v)
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="{SANS}">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}" stroke="#D9D4C7"/>')
        out.append(f'<text x="{L - 14}" y="{Y(v) + 8}" text-anchor="end" font-size="22" fill="#6B7385">{v:.2f}</text>')
    for tau in TAUS:
        out.append(f'<text x="{X(tau):.1f}" y="{H - B + 34}" text-anchor="middle" font-size="22" fill="#6B7385">{tau:g}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 14}" text-anchor="middle" font-size="24" fill="#3A4458">patience tau (moments of 0.4 s; log scale)</text>')
    out.append(f'<text x="28" y="{(T + H - B) / 2}" text-anchor="middle" font-size="24" fill="#3A4458" transform="rotate(-90 28 {(T + H - B) / 2})">P(walk on through the gust)</text>')
    for k, r in enumerate(SMALL):
        taus = [1.5 * (20 / 1.5) ** (i / 60) for i in range(61)]
        line = " ".join(f"{X(t):.1f},{Y(p_walk(t, r)):.1f}" for t in taus)
        out.append(f'<polyline points="{line}" fill="none" stroke="{COLOR[r]}" stroke-width="4"/>')
        for x in pts[str(r)]:
            out.append(f'<circle cx="{X(x["tau"]):.1f}" cy="{Y(x["p_walk"]):.1f}" r="9" fill="#F6F4EE" stroke="{COLOR[r]}" stroke-width="4"/>')
        best = max(pts[str(r)], key=lambda x: x["rate"])
        y0 = T + 70 + 150 * k
        out.append(f'<text x="{W - R + 20}" y="{y0}" font-size="26" fill="{COLOR[r]}">stop pays {r}</text>')
        out.append(f'<text x="{W - R + 20}" y="{y0 + 32}" font-size="22" fill="#3A4458">per moment: stop {r / 4:.3f}, walk {P / 10:.3f}</text>')
        out.append(f'<text x="{W - R + 20}" y="{y0 + 62}" font-size="22" fill="#3A4458">earns most at tau {best["tau"]:g} ({best["rate"]:.3f})</text>')
    out.append(f'<text x="{W - R + 20}" y="{T + 380}" font-size="22" fill="#6B7385">lines: the law; circles: measured</text>')
    out.append(f'<text x="{W - R + 20}" y="{T + 410}" font-size="22" fill="#6B7385">(1200 trials each, on the G1\'s world)</text>')
    out.append("</svg>")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args(argv)
    if a.run:
        run_all()
    pts = json.loads((TRACES / "patience.json").read_text(encoding="utf-8"))
    order = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))["order"]
    n = order.index("fig_patience") + 1
    notes = "; ".join(f"stop {r}: " + ", ".join(f"tau {x['tau']:g} {x['p_walk']:.2f} ({x['rate']:.3f}/moment)" for x in v) for r, v in pts.items())
    html = f"""<section id="fig_patience" data-transition="fade" style="background:#F6F4EE;color:#14213D;{SANS};padding:110px 128px 140px;display:flex;flex-direction:column;gap:24px">
<div style="display:flex;flex-direction:column;gap:12px">
<p style="font-size:24px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#B0521A">Part 2 · Measured</p>
<h2 style="font-size:56px;font-weight:700;line-height:1.1">Patience against the law, and what it earns</h2>
</div>
{svg(pts)}
<p style="font-size:26px;line-height:1.4;color:#3A4458">More patience, more walking on. Whether that pays depends on the deal: when stopping pays 0.4 the
impatient robot earns most; when it pays 0.1 the patient one does. Measured points sit a little under the law at high tau.</p>
<p style="position:absolute;left:128px;bottom:56px;width:1664px;font-size:24px;color:#6B7385">{FOOT} · {n}</p>
<aside>lab_patience.py serotonin --tau T --small R. {notes}. The shortfall at high tau: walking on pays 0 or 1, so its value is noisy, and a race in proportion to a noisy value picks it less often than the law, which uses the mean (Jensen).</aside>
</section>
"""
    (HERE / "slides" / "fig_patience.html").write_text(html, encoding="utf-8", newline="\n")
    print("wrote slides/fig_patience.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
