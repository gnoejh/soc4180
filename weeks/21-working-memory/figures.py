"""Draw the deck's measured figure: recall by age against the count law, at three fades.

    uv run python weeks/21-working-memory/figures.py          # redraw from traces/by_age.json
    uv run python weeks/21-working-memory/figures.py --run    # re-measure first (about 2 minutes)

--run runs lab_hold.py's 20 lists at each gamma in GAMMAS (in parallel) and keeps recall by
age in traces/by_age.json. The slide it writes, slides/fig_age.html, is a source of the
deck like the others; build.py assembles.
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
GAMMAS = (0.5, 0.75, 0.9)
COLOR = {0.5: "#B0521A", 0.75: "#2F6FB0", 0.9: "#14213D"}
SANS = "font-family:'IBM Plex Sans', Arial, sans-serif"
FOOT = "Holding a list"


def run_all() -> None:
    TRACES.mkdir(exist_ok=True)
    procs = [subprocess.Popen([sys.executable, str(HERE / "lab_hold.py"), "--lists", "20", "--gamma", str(g),
                               "--name", f"figures g{g}"], stdout=subprocess.DEVNULL) for g in GAMMAS]
    for p in procs:
        if p.wait():
            raise SystemExit("a run failed")
    rows = list(csv.DictReader(open(HERE / "results.csv", encoding="utf-8")))
    out = {}
    for g in GAMMAS:
        r = next(x for x in reversed(rows) if x["name"] == f"figures g{g}")
        out[str(g)] = {"by_age": [float(v) for v in r["by_age"].split()], "held": float(r["held"]), "law": float(r["law"])}
    (TRACES / "by_age.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def laws() -> dict[float, list[float]]:
    sys.path.insert(0, str(HERE))
    from lab_hold import law_by_position
    return {g: law_by_position(16, g, 64)[::-1] for g in GAMMAS}      # by age, 0 = newest


def svg(law: dict, meas: dict) -> str:
    W, H, L, R, T, B = 1660, 600, 130, 300, 30, 80
    X = lambda a: L + (W - L - R) * a / 15
    Y = lambda v: T + (H - T - B) * (1.0 - v)
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="{SANS}">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}" stroke="#D9D4C7"/>')
        out.append(f'<text x="{L - 14}" y="{Y(v) + 8}" text-anchor="end" font-size="22" fill="#6B7385">{v:.2f}</text>')
    for a in range(0, 16, 3):
        out.append(f'<text x="{X(a)}" y="{H - B + 34}" text-anchor="middle" font-size="22" fill="#6B7385">{a}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 14}" text-anchor="middle" font-size="24" fill="#3A4458">age of the push (pushes since; 0 = the newest)</text>')
    out.append(f'<text x="28" y="{(T + H - B) / 2}" text-anchor="middle" font-size="24" fill="#3A4458" transform="rotate(-90 28 {(T + H - B) / 2})">recalled as perceived</text>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(1 / 8)}" y2="{Y(1 / 8)}" stroke="#8FA3BF" stroke-dasharray="6 6"/>')
    for k, g in enumerate(GAMMAS):
        line = " ".join(f"{X(a):.1f},{Y(v):.1f}" for a, v in enumerate(law[g]))
        out.append(f'<polyline points="{line}" fill="none" stroke="{COLOR[g]}" stroke-width="4"/>')
        for a, v in enumerate(meas[str(g)]["by_age"]):
            out.append(f'<circle cx="{X(a):.1f}" cy="{Y(v):.1f}" r="8" fill="#F6F4EE" stroke="{COLOR[g]}" stroke-width="4"/>')
        m = meas[str(g)]
        out.append(f'<text x="{W - R + 20}" y="{T + 60 + 70 * k}" font-size="26" fill="{COLOR[g]}">gamma {g}</text>')
        out.append(f'<text x="{W - R + 20}" y="{T + 92 + 70 * k}" font-size="22" fill="#3A4458">held {m["held"]:.2f} (law {m["law"]:.2f})</text>')
    out.append(f'<text x="{W - R + 20}" y="{T + 300}" font-size="22" fill="#6B7385">lines: the law</text>')
    out.append(f'<text x="{W - R + 20}" y="{T + 330}" font-size="22" fill="#6B7385">circles: the G1, 20 lists</text>')
    out.append(f'<text x="{W - R + 20}" y="{Y(1 / 8) + 8}" font-size="22" fill="#6B7385">chance</text>')
    out.append("</svg>")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args(argv)
    if a.run:
        run_all()
    meas = json.loads((TRACES / "by_age.json").read_text(encoding="utf-8"))
    order = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))["order"]
    n = order.index("fig_age") + 1
    html = f"""<section id="fig_age" data-transition="fade" style="background:#F6F4EE;color:#14213D;{SANS};padding:110px 128px 140px;display:flex;flex-direction:column;gap:24px">
<div style="display:flex;flex-direction:column;gap:12px">
<p style="font-size:24px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#B0521A">Part 3 · Measured</p>
<h2 style="font-size:56px;font-weight:700;line-height:1.1">Recall by age: the newest are sharp, the oldest drown</h2>
</div>
{svg(laws(), meas)}
<p style="font-size:26px;line-height:1.4;color:#3A4458">A list of 16 pushes, d 64. A slower fade (gamma nearer 1) keeps older pushes, at the price of the
newest: at gamma 0.9 even the newest push is not certain.</p>
<p style="position:absolute;left:128px;bottom:56px;width:1664px;font-size:24px;color:#6B7385">{FOOT} · {n}</p>
<aside>lab_hold.py --lists 20 --gamma G. Each circle is 20 recalls, so a single point is +-0.1; the counts are the robust numbers: """ + \
        "; ".join(f"gamma {g}: {v['held']:.2f} held, law {v['law']:.2f}" for g, v in meas.items()) + """.</aside>
</section>
"""
    (HERE / "slides" / "fig_age.html").write_text(html, encoding="utf-8", newline="\n")
    print("wrote slides/fig_age.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
