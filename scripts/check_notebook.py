"""Check a generated lab.ipynb the way a student's Run All would exercise it.

    uv run scripts/check_notebook.py compile weeks/NN-slug/lab.ipynb
    uv run scripts/check_notebook.py run     weeks/NN-slug/lab.ipynb
    uv run scripts/check_notebook.py images  weeks/NN-slug/lab.ipynb OUTDIR

``compile`` byte-compiles every code cell, magics stripped. It is the only
check that touches an ``eval: false`` cell, which Quarto never executes -- a
broken f-string reached a student's runtime that way once.

``run`` compiles, then executes every cell in order with ``nbclient`` (cwd =
the notebook's directory), ``eval: false`` cells included. That is what Colab's
Run All does. Do not use it on week 10: its GPU training cell is the one
legitimate ``eval: false`` cell in the repo and would try to train on the CPU.

``images`` writes every image output to OUTDIR as PNG files, one per figure, so
a rendered deck's figures can be looked at without opening the deck. A figure
that renders but is wrong (clipped, overlapping labels, an autoscale that
missed a patch) is caught here and nowhere else.
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in ("compile", "run", "images"):
        print(__doc__)
        return 2
    cmd, nb_path = argv[1], Path(argv[2])
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    if cmd == "images":
        out = Path(argv[3]) if len(argv) > 3 else nb_path.parent / "_figures"
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for k, cell in enumerate(nb["cells"]):
            for o in cell.get("outputs", []):
                png = o.get("data", {}).get("image/png")
                if png is None:
                    continue
                if isinstance(png, list):
                    png = "".join(png)
                (out / f"{n:02d}_cell{k}.png").write_bytes(base64.b64decode(png))
                n += 1
        print(f"{n} images -> {out}")
        return 0

    for k, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # a magic becomes `pass` at the same indentation, so an `if` or `except`
        # whose whole body is a %pip line still compiles
        src = "\n".join(l[: len(l) - len(l.lstrip())] + "pass"
                        if l.lstrip().startswith(("%", "!")) else l
                        for l in src.split("\n"))
        compile(src, f"cell{k}", "exec")
    print("compile-check: every code cell OK")
    if cmd == "compile":
        return 0

    import nbformat
    from nbclient import NotebookClient

    node = nbformat.read(str(nb_path), as_version=4)
    client = NotebookClient(node, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(nb_path.parent)}})
    t0 = time.time()
    client.execute()
    print(f"nbclient: executed {len(node.cells)} cells in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
