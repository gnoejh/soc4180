"""lab_codecs: plug a trained model in -- CLIP as the brain's sense, checked by its certificate.

    uv run --extra vision weeks/26-codecs/lab_codecs.py words        # CLIP reads words, a sentence
    uv run --extra vision weeks/26-codecs/lab_codecs.py pictures     # CLIP sees the gallery's photos
    uv run --extra vision weeks/26-codecs/lab_codecs.py compare      # CLIP against other codecs
    uv run --extra vision weeks/26-codecs/lab_codecs.py capacity --d 64   # certificate against recall
    uv run --extra vision weeks/26-codecs/lab_codecs.py name         # name each picture with words
    uv run --extra vision weeks/26-codecs/lab_codecs.py words --sentence "a quick red fox jumps over the old wall"

The `vision` extra brings torch and transformers (`uv sync --extra vision` once; always run
with `uv run --extra vision`, or a plain `uv run` drops them). The CLIP weights
(openai/clip-vit-base-patch32, about 600 MB) are read from the Hugging Face cache; fetch
them once with --download.

The brain is untrained; vision and language come from a frozen model you plug in. They
meet through CODES: CLIP turns a word or a picture into 512 numbers, and the brain stores
codes (week 20's hippocampus, the `observer` profile). Two things decide whether that works:
  the cone       CLIP's raw codes all point roughly the same way: two unrelated words have
                 a cosine of about 0.8. Stored as they are, everything looks alike.
  centring       a Sense subtracts the vocabulary's mean code once (Sense.fit), and the
                 codes then behave almost like random white ones.
The CERTIFICATE (ubrain.certify) says, before anything is stored, how often an item of a
k-item episode comes back with these codes, beside the same number for ideal white codes.
    words      CLIP's text codes for about 170 words; a sentence heard and recalled
    pictures   CLIP's image codes for the 21 gallery photographs (the G1 walks past them in
               week 27); a sequence of pictures stored and recalled
    compare    the certificate of CLIP, of random codes (GaussianCodes) and of a
               bag-of-letters codec written here, on the same words
    capacity   the certificate beside measured recall: random k-word episodes (--episodes
               of each k), stored and recalled, at workspace --d
    name       each gallery picture's CLIP image code beside the text codes of the 21
               names: one space for both, so the nearest word names the picture
    --d D      the workspace (default 512, CLIP's own); try 64
    --raw      skip the centring: store CLIP's codes as they come, in their cone
    --download fetch the CLIP weights into the cache first (about 600 MB, once)

Every run appends rows to results.csv. ubrain is the course's local copy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np

import ubrain_local

ubrain_local.load()                 # the vendored wheel onto sys.path, before `import ubrain`

from ubrain import Brain, Channel, Manifest, profile  # noqa: E402
from ubrain.codecs import GaussianCodes, Sense, certify  # noqa: E402

HERE = Path(__file__).resolve().parent
GALLERY = ubrain_local.BODY / "scenes" / "gallery_cc0"
KS = (4, 8, 16)
WORDS = """apple banana orange grape lemon cherry bread cheese butter milk coffee tea water soup
rice egg fish chicken cake cookie dog cat horse cow sheep pig bird owl eagle duck fox wolf bear lion
tiger mouse rabbit frog snake bee butterfly tree flower grass leaf river lake sea mountain hill
forest desert island beach sun moon star cloud rain snow wind storm house door window roof wall
chair table bed lamp clock mirror cup plate knife spoon fork bottle box bag shoe hat coat shirt
dress glove ring watch phone book pen pencil paper letter map key coin car bus train bicycle boat
ship plane rocket road bridge tower castle church school library market garden park street city
village farm doctor teacher student king queen child mother father friend soldier farmer baker
red blue green yellow black white small large old young happy sad quick slow warm cold
run walk jump swim sing read write sleep eat drink open close build break find give""".split()
SENTENCE = "the old farmer saw a red fox near the river"


class Letters:
    """A codec written here: a word's code is the sum of a fixed random vector per letter
    (a bag of letters). Any object with these attributes and encode() is a codec."""
    name, modality, license = "bag-of-letters", "text", "course"

    def __init__(self, dim: int = 512, seed: int = 0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.table = {ch: rng.standard_normal(dim) for ch in "abcdefghijklmnopqrstuvwxyz"}
        self.digest = hashlib.sha256(f"letters{dim}{seed}".encode()).hexdigest()[:12]

    def encode(self, items) -> np.ndarray:
        return np.array([sum(self.table[c] for c in w if c in self.table) for w in items], dtype=float)


def cone(codec, items) -> tuple[float, float]:
    """Mean cosine between different items' codes, raw and after centring."""
    Z = np.asarray(codec.encode(list(items)), dtype=float)
    def mean_cos(M):
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        C = M @ M.T
        return float(C[~np.eye(len(M), dtype=bool)].mean())
    return mean_cos(Z), mean_cos(Z - Z.mean(0))


def observer(sense: Sense, d: int, V: int) -> Brain:
    body = Manifest("course/listener", "none", (
        Channel("camera", "codec", 1), Channel("recall", "event", 64), Channel("boundary", "event", 1),
        Channel("memory", "command", d), Channel("remembered", "choice", V)))
    return Brain(profile("observer"), body, codecs={"camera": sense}, rng=np.random.default_rng(0))


def make_sense(codec, items, d: int, raw: bool) -> Sense:
    """Sense.fit centres on the vocabulary's mean; raw=True builds the same sense with a zero mean."""
    if not raw:
        return Sense.fit(codec, items, d=d)
    zero = np.zeros(np.asarray(codec.encode(items[:1])).shape[1])
    vocabulary = np.asarray(Sense(codec, mean=zero, d=d, items=list(items)).raw(list(items)))
    return Sense(codec, mean=zero, d=d, items=list(items), vocabulary=vocabulary)


def log(rows: list[dict]) -> None:
    new = not (HERE / "results.csv").exists()
    with open(HERE / "results.csv", "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["when", "step", "codec", "items", "cone_raw", "cone_centred",
                                           "certificate", "recalled"])
        if new:
            wr.writeheader()
        for r in rows:
            wr.writerow({"when": datetime.now().isoformat(timespec="seconds"), **r})


def certificate_line(codec, items, d) -> tuple[str, list[float]]:
    cert = certify(codec, list(items), d=d, ks=KS, n_trials=2000)
    return cert.summary(), list(cert.accuracy)


def step_words(a) -> None:
    from ubrain.extras.clip import CLIP
    clip = CLIP("text")
    sentence = a.sentence.split()
    vocab = sorted(set(WORDS) | set(sentence))
    raw, cen = cone(clip, vocab)
    print(f"CLIP's text codes for {len(vocab)} words: mean cosine between different words {raw:.2f} raw, "
          f"{cen:.2f} centred\ncertifying (about half a minute) ...", flush=True)
    summary, acc = certificate_line(clip, vocab, a.d)
    print(summary)
    sense = make_sense(clip, vocab, a.d, a.raw)
    brain = observer(sense, a.d, len(vocab))
    for w in sentence:
        brain.tick({"camera": w})
    recalled = [vocab[brain.tick({"recall": i})["remembered"]] for i in range(len(sentence))]
    right = sum(r == w for r, w in zip(recalled, sentence))
    print(f"\nheard   : {' '.join(sentence)}\nrecalled: {' '.join(recalled)}   ({right} of {len(sentence)})")
    if a.raw:
        print("(raw: stored without centring; the certificate above is for centred codes)")
    log([{"step": "words", "codec": "CLIP text raw" if a.raw else "CLIP text", "items": len(vocab), "cone_raw": round(raw, 3),
          "cone_centred": round(cen, 3), "certificate": " ".join(f"{x:.3f}" for x in acc),
          "recalled": f"{right}/{len(sentence)}"}])


def step_pictures(a) -> None:
    from ubrain.extras.clip import CLIP
    clip = CLIP("image")
    pictures = sorted(GALLERY.glob("*.png"))
    names = [p.stem.split("_", 1)[1] for p in pictures]
    raw, cen = cone(clip, pictures)
    print(f"CLIP's image codes for the {len(pictures)} gallery photographs ({', '.join(names)}):\n"
          f"mean cosine between different pictures {raw:.2f} raw, {cen:.2f} centred\ncertifying ...", flush=True)
    summary, acc = certificate_line(clip, pictures, a.d)
    print(summary)
    sense = make_sense(clip, pictures, a.d, a.raw)
    order = np.random.default_rng(a.seed).permutation(len(pictures))[:a.k]
    brain = observer(sense, a.d, len(pictures))
    for i in order:
        brain.tick({"camera": pictures[int(i)]})
    recalled = [int(brain.tick({"recall": j})["remembered"]) for j in range(len(order))]
    right = sum(int(r) == int(i) for r, i in zip(recalled, order))
    print(f"\nshown   : {' '.join(names[int(i)] for i in order)}\nrecalled: {' '.join(names[r] for r in recalled)}"
          f"   ({right} of {len(order)})")
    if a.raw:
        print("(raw: stored without centring; the certificate above is for centred codes)")
    log([{"step": "pictures", "codec": "CLIP image raw" if a.raw else "CLIP image", "items": len(pictures), "cone_raw": round(raw, 3),
          "cone_centred": round(cen, 3), "certificate": " ".join(f"{x:.3f}" for x in acc),
          "recalled": f"{right}/{len(order)}"}])


def step_compare(a) -> None:
    from ubrain.extras.clip import CLIP
    vocab = sorted(set(WORDS))
    codecs = [("CLIP text", CLIP("text"), vocab), ("random (GaussianCodes)", GaussianCodes(dim=512), list(range(len(vocab)))),
              ("bag of letters (written here)", Letters(512), vocab)]
    print(f"the same {len(vocab)} words, three codecs; recall predicted for a k-word episode, d {a.d}")
    print("  codec                            cone raw  centred   k=4    k=8    k=16")
    rows = []
    for name, codec, items in codecs:
        raw, cen = cone(codec, items)
        acc = list(certify(codec, items, d=a.d, ks=KS, n_trials=2000).accuracy)
        print(f"  {name:32s} {raw:6.2f}   {cen:6.2f}  " + "  ".join(f"{x:.3f}" for x in acc), flush=True)
        rows.append({"step": "compare", "codec": name, "items": len(items), "cone_raw": round(raw, 3),
                     "cone_centred": round(cen, 3), "certificate": " ".join(f"{x:.3f}" for x in acc), "recalled": ""})
    log(rows)


def step_capacity(a) -> None:
    from ubrain.extras.clip import CLIP
    clip = CLIP("text")
    vocab = sorted(set(WORDS))
    ks = (2, 4, 8, 16, 32)
    acc = list(certify(clip, vocab, d=a.d, ks=ks, n_trials=2000).accuracy)
    sense = make_sense(clip, vocab, a.d, a.raw)
    rng = np.random.default_rng(a.seed)
    print(f"CLIP text, {len(vocab)} words, d {a.d}{' raw' if a.raw else ''}: {a.episodes} episodes of k random words each")
    print("     k   certificate   measured recall")
    rows = []
    for k, c in zip(ks, acc):
        right = 0
        for _ in range(a.episodes):
            words = [vocab[int(i)] for i in rng.choice(len(vocab), k, replace=False)]
            brain = observer(sense, a.d, len(vocab))
            for w in words:
                brain.tick({"camera": w})
            right += sum(vocab[brain.tick({"recall": i})["remembered"]] == w for i, w in enumerate(words))
        m = right / (k * a.episodes)
        print(f"  {k:4d}      {c:.3f}          {m:.3f}", flush=True)
        rows.append({"step": "capacity", "codec": f"CLIP text d {a.d}" + (" raw" if a.raw else ""), "items": k,
                     "cone_raw": "", "cone_centred": "", "certificate": f"{c:.3f}", "recalled": f"{m:.3f}"})
    log(rows)


def step_name(a) -> None:
    from ubrain.extras.clip import CLIP
    pictures = sorted(GALLERY.glob("*.png"))
    names = [p.stem.split("_", 1)[1] for p in pictures]
    unit = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
    I = unit(np.asarray(CLIP("image").encode(pictures), dtype=float))
    T = unit(np.asarray(CLIP("text").encode(names), dtype=float))
    S = I @ T.T                                    # cosine of each picture to each word
    right = 0
    print("picture       nearest word   cosine   (its own name: cosine)")
    for i, n in enumerate(names):
        j = int(np.argmax(S[i]))
        right += j == i
        print(f"  {n:11s} {names[j]:12s}  {S[i, j]:.3f}   ({S[i, i]:.3f}){'' if j == i else '   WRONG'}")
    print(f"\n{right} of {len(names)} pictures named by the nearest word; random codes would name about 1 by chance")
    log([{"step": "name", "codec": "CLIP image->text", "items": len(names), "cone_raw": "", "cone_centred": "",
          "certificate": "", "recalled": f"{right}/{len(names)}"}])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.split("\n")[7:]))
    ap.add_argument("step", choices=("words", "pictures", "compare", "capacity", "name"))
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--raw", action="store_true", help="store the codes without centring")
    ap.add_argument("--sentence", default=SENTENCE)
    ap.add_argument("--k", type=int, default=8, help="pictures: how many to show in order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=20, help="capacity: episodes per k")
    ap.add_argument("--download", action="store_true", help="fetch the CLIP weights first (~600 MB, once)")
    a = ap.parse_args(argv)
    if a.download:
        from huggingface_hub import snapshot_download
        snapshot_download("openai/clip-vit-base-patch32")
    {"words": step_words, "pictures": step_pictures, "compare": step_compare, "capacity": step_capacity, "name": step_name}[a.step](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
