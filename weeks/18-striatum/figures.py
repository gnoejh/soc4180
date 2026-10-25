"""Draw the deck's two measured figures from lab_choose.py's trial logs, as inline SVG slides.

    uv run python weeks/18-striatum/figures.py          # redraw from traces/*.json
    uv run python weeks/18-striatum/figures.py --run    # re-measure first (about 5 minutes, 21 runs in parallel)

--run runs lab_choose.py 300 trials at each learning rate in ETAS with seeds 0-3, plus
the values trace (eta 0.05, seed 1), and keeps each run's trial log in traces/. The
slides it writes, slides/fig_values.html and slides/fig_eta.html, are sources of the
deck like the others; build.py assembles.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRACES = HERE / "traces"
ETAS = (0.05, 0.2, 0.5, 0.8, 1.0)
SEEDS = (0, 1, 2, 3)
WORLD = np.array([[0.85, 0.25, 0.55], [0.475, 0.85, 0.55]])     # measured with --map
MATCHING = float((WORLD * WORLD / WORLD.sum(1, keepdims=True)).sum(1).mean())
SANS = "font-family:'IBM Plex Sans', Arial, sans-serif"
FOOT = "Learning from reward"
MOVE_COLOR = ("#2F6FB0", "#B0521A", "#8FA3BF")
MOVES = ("brake early", "brake late", "keep walking")


def run_all() -> None:
    TRACES.mkdir(exist_ok=True)
    jobs = [(eta, seed, f"traces/eta_{eta}_s{seed}.json") for eta in ETAS for seed in SEEDS]
    jobs.append((0.05, 1, "traces/values.json"))
    procs = [subprocess.Popen([sys.executable, str(HERE / "lab_choose.py"), "--trials", "300",
                               "--eta", str(eta), "--seed", str(seed), "--name", "figures", "--log", log],
                              stdout=subprocess.DEVNULL) for eta, seed, log in jobs]
    for p in procs:
        if p.wait():
            raise SystemExit("a run failed")
    for _eta, _seed, log in jobs:                       # keep only what the figures read, compactly
        path = HERE / log
        d = json.loads(path.read_text(encoding="utf-8"))
        if path.name != "values.json":
            for t in d["trials_log"]:
                t.pop("Q_before_outcome", None)
        path.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")


def load(name: str) -> list[dict]:
    return json.loads((TRACES / name).read_text(encoding="utf-8"))["trials_log"]


def last_third(log: list[dict]) -> tuple[float, float]:
    """(share stayed up, share of trials where the right brake was chosen) in the last third."""
    part = log[len(log) - len(log) // 3:]
    up = np.mean([t["up"] for t in part])
    right = np.mean([t["move"] == t["cue"] for t in part])        # soon -> 0, late -> 1
    return float(up), float(right)


def svg_values(log: list[dict]) -> str:
    W, H, L, R, T, B = 1660, 600, 130, 260, 30, 80
    n = len(log)
    X = lambda i: L + (W - L - R) * i / (n - 1)
    Y = lambda v: T + (H - T - B) * (1.0 - v)
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="{SANS}">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}" stroke="#D9D4C7"/>')
        out.append(f'<text x="{L - 14}" y="{Y(v) + 8}" text-anchor="end" font-size="22" fill="#6B7385">{v:.2f}</text>')
    for i in range(0, n + 1, 50):
        out.append(f'<text x="{X(min(i, n - 1))}" y="{H - B + 34}" text-anchor="middle" font-size="22" fill="#6B7385">{i}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 14}" text-anchor="middle" font-size="24" fill="#3A4458">trial</text>')
    out.append(f'<text x="28" y="{(T + H - B) / 2}" text-anchor="middle" font-size="24" fill="#3A4458" transform="rotate(-90 28 {(T + H - B) / 2})">value Q, cue "soon"</text>')
    for a in range(3):
        y = Y(WORLD[0, a])
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y}" y2="{y}" stroke="{MOVE_COLOR[a]}" stroke-width="2" stroke-dasharray="10 8"/>')
        pts = " ".join(f"{X(i):.1f},{Y(t['Q_before_outcome'][0][a]):.1f}" for i, t in enumerate(log))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{MOVE_COLOR[a]}" stroke-width="4"/>')
        out.append(f'<text x="{W - R + 16}" y="{y + 8}" font-size="24" fill="{MOVE_COLOR[a]}">{MOVES[a]} {WORLD[0, a]:.2f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg_eta(rows: list[tuple[float, float, float]]) -> str:
    """rows: (eta, mean stayed up, mean right-brake share): grouped bars."""
    W, H, L, R, T, B = 1660, 600, 130, 40, 30, 90
    Y = lambda v: T + (H - T - B) * (1.0 - v)
    gw = (W - L - R) / len(rows)
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="{SANS}">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}" stroke="#D9D4C7"/>')
        out.append(f'<text x="{L - 14}" y="{Y(v) + 8}" text-anchor="end" font-size="22" fill="#6B7385">{v:.2f}</text>')
    for v, label, color in ((0.85, "always the right brake 0.85", "#14213D"), (MATCHING, f"matching {MATCHING:.2f}", "#B0521A")):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}" stroke="{color}" stroke-width="2" stroke-dasharray="10 8"/>')
    for k, (label, color) in enumerate(((f"matching {MATCHING:.2f}", "#B0521A"), ("always the right brake 0.85", "#14213D"))):
        x = L + 10 + 330 * k
        out.append(f'<line x1="{x}" x2="{x + 40}" y1="{H - 25}" y2="{H - 25}" stroke="{color}" stroke-width="3" stroke-dasharray="10 6"/>')
        out.append(f'<text x="{x + 50}" y="{H - 18}" font-size="22" fill="#3A4458">{label}</text>')
    for k, (eta, up, right) in enumerate(rows):
        x0 = L + k * gw + gw * 0.18
        for j, (v, c) in enumerate(((right, "#2F6FB0"), (up, "#E07A2F"))):
            x = x0 + j * gw * 0.33
            out.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{gw * 0.3:.1f}" height="{Y(0) - Y(v):.1f}" fill="{c}"/>')
            out.append(f'<text x="{x + gw * 0.15:.1f}" y="{Y(v) - 10:.1f}" text-anchor="middle" font-size="22" fill="#14213D">{v:.2f}</text>')
        out.append(f'<text x="{L + k * gw + gw / 2:.1f}" y="{H - B + 36}" text-anchor="middle" font-size="26" fill="#14213D">eta {eta:g}</text>')
    out.append(f'<rect x="{W - R - 560}" y="{H - 34}" width="24" height="18" fill="#2F6FB0"/><text x="{W - R - 526}" y="{H - 18}" font-size="22" fill="#3A4458">right brake chosen</text>')
    out.append(f'<rect x="{W - R - 290}" y="{H - 34}" width="24" height="18" fill="#E07A2F"/><text x="{W - R - 256}" y="{H - 18}" font-size="22" fill="#3A4458">stayed up</text>')
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

    log = load("values.json")
    slide("fig_values", order.index("fig_values") + 1, "Part 3 · Measured",
          "The values become the world's payoffs (cue \"soon\", eta 0.05)",
          svg_values(log),
          "Each line is one value Q, trial by trial; dashed, what the world pays (measured with --map). "
          "A slow learner (eta 0.05) ends close to every payoff, from nothing but staying up or falling.",
          "lab_choose.py --trials 300 --eta 0.05 --seed 1. The value of a move changes only on the trials that choose it, which is why 'brake late' moves least: it is rarely chosen once its value has dropped. Mean |Q - world| at eta 0.05 over 4 seeds: 0.06.")

    rows = []
    for eta in ETAS:
        res = [last_third(load(f"eta_{eta}_s{seed}.json")) for seed in SEEDS]
        rows.append((eta, float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res]))))
        print(f"eta {eta}: stayed up {rows[-1][1]:.3f}, right brake {rows[-1][2]:.2f}")
    slide("fig_eta", order.index("fig_eta") + 1, "Part 3 · Measured",
          "Accurate values are not the best choices",
          svg_eta(rows),
          "Last third of 300 trials, mean of 4 seeds. Slow learners know the world best and choose by "
          "matching; fast learners know it worse and choose better, until eta 1 keeps only the last outcome.",
          "lab_choose.py --trials 300 --eta E --seed 0..3. At eta 0.05 and 0.2 the choices sit at matching (0.66). A fast learner drops a move's value to near zero after one fall, so the race becomes nearly exclusive (win-stay, lose-shift). At eta 1 every move that worked last time has value 1: ties again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
