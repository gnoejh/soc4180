"""Draw the deck's two measured figures from lab_adapt.py's traces, as inline SVG slides.

    uv run python weeks/17-adaptation/figures.py          # redraw from traces/*.json
    uv run python weeks/17-adaptation/figures.py --run    # re-measure first (about 3 minutes)

--run runs lab_adapt.py for each schedule and learner below and keeps each run's
last_run.json in traces/. The slides it writes, slides/fig_basic.html and
slides/fig_recovery.html, are sources of the deck like the others; build.py assembles.
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
RUNS = [("basic", "off"), ("basic", "single"), ("basic", "dual"),
        ("recovery", "single"), ("recovery", "dual")]
COLOR = {"off": "#8FA3BF", "single": "#2F6FB0", "dual": "#E07A2F"}
LABEL = {"off": "no learning", "single": "one rate", "dual": "two rates"}
SANS = "font-family:'IBM Plex Sans', Arial, sans-serif"
FOOT = "Walking in a side wind"


def run_all() -> None:
    TRACES.mkdir(exist_ok=True)
    for schedule, learner in RUNS:
        print(f"measuring {schedule} / {learner} ...", flush=True)
        subprocess.run([sys.executable, str(HERE / "lab_adapt.py"), "--schedule", schedule,
                        "--learner", learner, "--name", "figures"], check=True,
                       stdout=subprocess.DEVNULL)
        shutil.copy(HERE / "last_run.json", TRACES / f"{schedule}_{learner}.json")


def load(schedule: str, learner: str) -> list[dict]:
    return json.loads((TRACES / f"{schedule}_{learner}.json").read_text(encoding="utf-8"))["strides"]


def chart(series, key, ylo, yhi, ylabel, spans, marks=()) -> str:
    """An SVG line chart, 1660 x 640: series = [(learner, strides)], spans = [(x0, x1, label, fill)]."""
    W, H, L, R, T, B = 1660, 640, 130, 30, 30, 80
    n = max(len(s) for _, s in series)
    X = lambda i: L + (W - L - R) * (i - 0.5) / n                       # stride i (1-based), centred
    Y = lambda v: T + (H - T - B) * (yhi - v) / (yhi - ylo)
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="{SANS}">']
    for x0, x1, label, fill in spans:
        xa, xb = L + (W - L - R) * (x0 - 1) / n, L + (W - L - R) * x1 / n
        out.append(f'<rect x="{xa:.1f}" y="{T}" width="{xb - xa:.1f}" height="{H - T - B}" fill="{fill}"/>')
        out.append(f'<text x="{(xa + xb) / 2:.1f}" y="{T + 34}" text-anchor="middle" font-size="24" fill="#6B7385">{label}</text>')
    step = 0.05
    v = round(ylo / step) * step
    while v <= yhi + 1e-9:
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{"#14213D" if abs(v) < 1e-9 else "#D9D4C7"}" stroke-width="{2 if abs(v) < 1e-9 else 1}"/>')
        out.append(f'<text x="{L - 14}" y="{Y(v) + 8:.1f}" text-anchor="end" font-size="22" fill="#6B7385">{v:+.2f}</text>')
        v += step
    for i in range(0, n + 1, 10):
        if i:
            out.append(f'<text x="{X(i):.1f}" y="{H - B + 34}" text-anchor="middle" font-size="22" fill="#6B7385">{i}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 14}" text-anchor="middle" font-size="24" fill="#3A4458">stride (0.8 s each)</text>')
    out.append(f'<text x="28" y="{(T + H - B) / 2}" text-anchor="middle" font-size="24" fill="#3A4458" transform="rotate(-90 28 {(T + H - B) / 2})">{ylabel}</text>')
    for x, learner in marks:
        out.append(f'<line x1="{L + (W - L - R) * (x - 1) / n:.1f}" x2="{L + (W - L - R) * (x - 1) / n:.1f}" y1="{T}" y2="{H - B}" stroke="{COLOR[learner]}" stroke-width="2" stroke-dasharray="8 6"/>')
    for learner, strides in series:
        pts = " ".join(f"{X(i + 1):.1f},{Y(max(min(s[key], yhi), ylo)):.1f}" for i, s in enumerate(strides))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{COLOR[learner]}" stroke-width="4" stroke-linejoin="round"/>')
    for k, (learner, _s) in enumerate(series):
        y = T + 70 + 36 * k
        out.append(f'<line x1="{W - R - 300}" x2="{W - R - 250}" y1="{y}" y2="{y}" stroke="{COLOR[learner]}" stroke-width="5"/>')
        out.append(f'<text x="{W - R - 238}" y="{y + 8}" font-size="24" fill="#14213D">{LABEL[learner]}</text>')
    out.append("</svg>")
    return "\n".join(out)


def slide(sid: str, n: int, kicker: str, title: str, svg: str, caption: str, notes: str) -> None:
    html = f"""<section id="{sid}" data-transition="fade" style="background:#F6F4EE;color:#14213D;{SANS};padding:110px 128px 140px;display:flex;flex-direction:column;gap:24px">
<div style="display:flex;flex-direction:column;gap:12px">
<p style="font-size:24px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#B0521A">{kicker}</p>
<h2 style="font-size:56px;font-weight:700;line-height:1.1">{title}</h2>
</div>
{svg}
<p style="font-size:26px;line-height:1.4;color:#3A4458">{caption}</p>
<p style="position:absolute;left:128px;bottom:56px;width:1664px;font-size:24px;color:#6B7385">{FOOT} · {n}</p>
<aside>{notes}</aside>
</section>
"""
    (HERE / "slides" / f"{sid}.html").write_text(html, encoding="utf-8", newline="\n")
    print(f"wrote slides/{sid}.html")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true", help="re-measure the traces first")
    a = ap.parse_args(argv)
    if a.run:
        run_all()
    order = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))["order"]

    basic = [(lr, load("basic", lr)) for lr in ("off", "single", "dual")]
    slide("fig_basic", order.index("fig_basic") + 1, "Part 3 · Measured", "The turn, stride by stride",
          chart(basic, "turn", -0.15, 0.20, "turn (rad/s, + left)",
                [(6, 35, "wind 15 N", "#FCEBDD")]),
          "No learning keeps turning while the wind blows. Both learners bring the turn down, and both "
          "turn the other way when the wind stops: the after-effect.",
          "lab_adapt.py --schedule basic: 5 calm strides, 30 with the wind (shaded), 30 calm. One point is one stride's mean yaw rate. Ask the class to find the after-effect before you point at it.")

    rec = [(lr, load("recovery", lr)) for lr in ("single", "dual")]
    marks = []
    for lr, strides in rec:
        first_clamped = next(i for i, s in enumerate(strides) if s["clamped"]) + 1
        marks.append((first_clamped, lr))
    slide("fig_recovery", order.index("fig_recovery") + 1, "Part 3 · Measured",
          "The learned counter-turn, through unlearning and the clamp",
          chart(rec, "counter", -0.20, 0.15, "counter-turn (rad/s)",
                [(6, 45, "wind 15 N", "#FCEBDD")], marks),
          "After the counter-wind both learners are back near zero (dashed: the clamp starts). "
          "Clamped, one rate stays there; two rates drift back toward the adapted side.",
          "lab_adapt.py --schedule recovery. The counter-wind block runs until the counter-turn crosses zero: 4 strides for two rates, 6 for one rate, so the dashed lines differ. The drift back during the clamp is spontaneous recovery: the fast state forgets the counter-wind, the slow state still holds the wind.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
