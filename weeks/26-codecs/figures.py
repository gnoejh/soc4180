"""Draw the deck's measured figure: the certificate against measured recall, for four workspaces.

    uv run --extra vision weeks/26-codecs/figures.py          # redraw from traces/capacity.json
    uv run --extra vision weeks/26-codecs/figures.py --run    # re-measure first (about 2 minutes)

--run runs lab_codecs.py capacity at each workspace in DS (and once raw, at 512), and keeps
the certificate and the measured recall in traces/capacity.json. The slide it writes,
slides/fig_capacity.html, is a source of the deck like the others.
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
DS = (512, 128, 64, 32)
KS = (2, 4, 8, 16, 32)
COLOR = {"512": "#14213D", "128": "#2F6FB0", "64": "#E07A2F", "32": "#B0521A", "512 raw": "#C0392B"}
SANS = "font-family:'IBM Plex Sans', Arial, sans-serif"
FOOT = "Plugging in a trained model"


def run_all() -> None:
    TRACES.mkdir(exist_ok=True)
    for d in DS:
        subprocess.run([sys.executable, str(HERE / "lab_codecs.py"), "capacity", "--d", str(d)],
                       check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(HERE / "lab_codecs.py"), "capacity", "--raw"],
                   check=True, stdout=subprocess.DEVNULL)
    out: dict[str, dict] = {}
    for row in csv.DictReader(open(HERE / "results.csv", encoding="utf-8")):
        if row["step"] == "capacity":
            key = row["codec"].replace("CLIP text d ", "")
            out.setdefault(key, {})[row["items"]] = {"certificate": float(row["certificate"]),
                                                     "measured": float(row["recalled"])}
    (TRACES / "capacity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def svg(pts: dict) -> str:
    W, H, L, R, T, B = 1660, 600, 130, 420, 30, 80
    X = lambda k: L + (W - L - R) * (math.log2(k) - 1) / (math.log2(32) - 1)
    Y = lambda v: T + (H - T - B) * (1.0 - v)
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="{SANS}">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}" stroke="#D9D4C7"/>')
        out.append(f'<text x="{L - 14}" y="{Y(v) + 8}" text-anchor="end" font-size="22" fill="#6B7385">{v:.2f}</text>')
    for k in KS:
        out.append(f'<text x="{X(k):.1f}" y="{H - B + 34}" text-anchor="middle" font-size="22" fill="#6B7385">{k}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 14}" text-anchor="middle" font-size="24" fill="#3A4458">words in the episode, k (log scale)</text>')
    out.append(f'<text x="28" y="{(T + H - B) / 2}" text-anchor="middle" font-size="24" fill="#3A4458" transform="rotate(-90 28 {(T + H - B) / 2})">fraction recalled</text>')
    for i, key in enumerate([str(d) for d in DS] + ["512 raw"]):
        if key not in pts:
            continue
        col, row = COLOR[key], pts[key]
        if key != "512 raw":
            line = " ".join(f"{X(k):.1f},{Y(row[str(k)]['certificate']):.1f}" for k in KS)
            out.append(f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="4"/>')
        else:
            line = " ".join(f"{X(k):.1f},{Y(row[str(k)]['measured']):.1f}" for k in KS)
            out.append(f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="3" stroke-dasharray="10 8"/>')
        for k in KS:
            out.append(f'<circle cx="{X(k):.1f}" cy="{Y(row[str(k)]["measured"]):.1f}" r="9" fill="#F6F4EE" stroke="{col}" stroke-width="4"/>')
        label = f"d {key}" if key != "512 raw" else "d 512, not centred"
        out.append(f'<text x="{W - R + 20}" y="{T + 60 + 48 * i}" font-size="26" fill="{col}">{label}</text>')
    out.append(f'<text x="{W - R + 20}" y="{T + 330}" font-size="22" fill="#6B7385">lines: the certificate</text>')
    out.append(f'<text x="{W - R + 20}" y="{T + 360}" font-size="22" fill="#6B7385">circles: measured, 20 episodes</text>')
    out.append(f'<text x="{W - R + 20}" y="{T + 390}" font-size="22" fill="#6B7385">dashed: measured only (the</text>')
    out.append(f'<text x="{W - R + 20}" y="{T + 420}" font-size="22" fill="#6B7385">certificate assumes centring)</text>')
    out.append("</svg>")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args(argv)
    if a.run:
        run_all()
    pts = json.loads((TRACES / "capacity.json").read_text(encoding="utf-8"))
    order = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))["order"]
    n = order.index("fig_capacity") + 1
    notes = "; ".join(f"d {key}: " + ", ".join(f"k {k} {v['certificate']:.2f}/{v['measured']:.2f}" for k, v in row.items())
                      for key, row in pts.items())
    html = f"""<section id="fig_capacity" data-transition="fade" style="background:#F6F4EE;color:#14213D;{SANS};padding:110px 128px 140px;display:flex;flex-direction:column;gap:24px">
<div style="display:flex;flex-direction:column;gap:12px">
<p style="font-size:24px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#B0521A">Part 2 · Measured</p>
<h2 style="font-size:56px;font-weight:700;line-height:1.1">The certificate, said before; the recall, measured after</h2>
</div>
{svg(pts)}
<p style="font-size:26px;line-height:1.4;color:#3A4458">For centred CLIP codes the certificate predicts recall at every workspace, before a single word is stored.
Skip the centring and the same codes in the full workspace recall worse than d 128 does: the cone, not the size, is the limit.</p>
<p style="position:absolute;left:128px;bottom:56px;width:1664px;font-size:24px;color:#6B7385">{FOOT} · {n}</p>
<aside>lab_codecs.py capacity --d D (and --raw). Certificate / measured: {notes}.</aside>
</section>
"""
    (HERE / "slides" / "fig_capacity.html").write_text(html, encoding="utf-8", newline="\n")
    print("wrote slides/fig_capacity.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
