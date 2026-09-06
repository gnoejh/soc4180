"""Assemble the GitHub Pages site from rendered week directories.

`quarto render` leaves `slides.html` and `lab.ipynb` beside each `slides.qmd`.
This collects them into `_site/` and writes a landing page, so the decks are
reachable from a browser alone -- which is the point: `slides.html` is a
gitignored build artifact (3-5 MB each, resources inlined), so the web copy is
the only shared copy.

Weeks whose deck failed to render are listed as unavailable rather than
silently dropped, and this script always exits 0: the render step owns failure.
"""

from __future__ import annotations

import argparse
import html
import pathlib
import re
import shutil

REPO = "gnoejh/soc4180"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/main/weeks"

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _frontmatter(qmd: pathlib.Path) -> tuple[str, str]:
    """Title and subtitle from the YAML header, without needing a YAML parser."""
    text = qmd.read_text(encoding="utf-8")
    head = text.split("---", 2)[1] if text.startswith("---") else text[:2000]

    def field(name: str) -> str:
        m = re.search(rf'^{name}:\s*"?(.*?)"?\s*$', head, re.MULTILINE)
        return m.group(1) if m else ""

    return field("title") or qmd.parent.name, field("subtitle")


def build(out: pathlib.Path) -> list[dict]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / ".nojekyll").write_text("", encoding="utf-8")

    weeks = []
    for week_dir in sorted((ROOT / "weeks").iterdir()):
        qmd = week_dir / "slides.qmd"
        if not qmd.is_file():
            continue
        slug = week_dir.name
        title, subtitle = _frontmatter(qmd)
        dest = out / slug
        dest.mkdir()

        deck = week_dir / "slides.html"
        if deck.is_file():
            shutil.copy2(deck, dest / "slides.html")
        lab = week_dir / "lab.ipynb"
        if lab.is_file():
            shutil.copy2(lab, dest / "lab.ipynb")

        weeks.append(
            {
                "slug": slug,
                "title": title,
                "subtitle": subtitle,
                "deck": deck.is_file(),
                "lab": lab.is_file(),
            }
        )
    (out / "index.html").write_text(_index(weeks), encoding="utf-8")
    return weeks


def _card(w: dict) -> str:
    num = w["slug"][1:3]
    title = html.escape(w["title"])
    subtitle = html.escape(w["subtitle"])
    links = []
    if w["deck"]:
        links.append(f'<a class="primary" href="{w["slug"]}/slides.html">Slides</a>')
    else:
        links.append('<span class="missing">Slides unavailable</span>')
    if w["lab"]:
        links.append(f'<a href="{COLAB}/{w["slug"]}/lab.ipynb">Open lab in Colab</a>')
        links.append(f'<a href="{w["slug"]}/lab.ipynb" download>Download notebook</a>')
    return f"""      <article class="week">
        <div class="num">{num}</div>
        <div class="body">
          <h2>{title}</h2>
          <p class="sub">{subtitle}</p>
          <nav>{" ".join(links)}</nav>
        </div>
      </article>"""


def _index(weeks: list[dict]) -> str:
    cards = "\n".join(_card(w) for w in weeks)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SOC4180 - Robot and AI</title>
<style>
  :root {{
    --bg: #fbfaf8; --fg: #1c1b19; --muted: #6b6862; --line: #e2ded7;
    --card: #ffffff; --accent: #7a4b2a;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #171614; --fg: #ece9e3; --muted: #9c968c; --line: #2e2c28;
      --card: #201e1b; --accent: #d59a6a;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--fg);
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  }}
  .wrap {{ max-width: 62rem; margin: 0 auto; padding: 4rem 1.5rem 6rem; }}
  header {{ border-bottom: 1px solid var(--line); padding-bottom: 2rem; margin-bottom: 2.5rem; }}
  h1 {{ font-size: 2rem; margin: 0 0 .5rem; letter-spacing: -.02em; }}
  header p {{ margin: 0; color: var(--muted); max-width: 40rem; }}
  .week {{
    display: flex; gap: 1.25rem; align-items: baseline;
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 1.1rem 1.35rem; margin-bottom: .75rem;
  }}
  .num {{
    font-variant-numeric: tabular-nums; font-weight: 600; font-size: 1.1rem;
    color: var(--accent); min-width: 2.2rem;
  }}
  .body {{ flex: 1; min-width: 0; }}
  h2 {{ font-size: 1.05rem; margin: 0 0 .15rem; font-weight: 600; }}
  .sub {{ margin: 0 0 .6rem; color: var(--muted); font-size: .9rem; }}
  nav {{ display: flex; flex-wrap: wrap; gap: .4rem 1rem; font-size: .875rem; }}
  nav a {{ color: var(--accent); text-decoration: none; }}
  nav a:hover {{ text-decoration: underline; }}
  nav a.primary {{ font-weight: 600; }}
  .missing {{ color: var(--muted); font-style: italic; }}
  footer {{ margin-top: 3rem; color: var(--muted); font-size: .85rem; }}
  footer a {{ color: var(--accent); }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>SOC4180 &mdash; Robot and AI</h1>
      <p>Robotics and robot learning, taught entirely in simulation. The
      through-line is a walking humanoid: classical control first, learned
      control second. Slides open in the browser; labs open in Colab with
      nothing to install.</p>
    </header>
{cards}
    <footer>
      Built from <a href="https://github.com/{REPO}">github.com/{REPO}</a>.
      Set the Colab runtime to <strong>T4 GPU</strong> before running a lab
      &mdash; MuJoCo renders through EGL and needs it.
    </footer>
  </div>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "_site"), help="output directory")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    weeks = build(out)

    print(f"site -> {out}")
    for w in weeks:
        deck = "deck" if w["deck"] else "NO DECK"
        lab = "lab" if w["lab"] else "no lab"
        print(f"  {w['slug']:<24} {deck:<8} {lab}")
    missing = [w["slug"] for w in weeks if not w["deck"]]
    print(f"\n{len(weeks) - len(missing)}/{len(weeks)} decks included")
    if missing:
        print("missing decks: " + ", ".join(missing))


if __name__ == "__main__":
    main()
