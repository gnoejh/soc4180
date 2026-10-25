# Lab 3 -- Plugging in a trained model: codecs and certificates

**Goal:** use a frozen AI model as the brain's sense, and check what its codes do to
memory before relying on them.

## Run

```bash
uv run --extra vision examples/03_clip_words.py
```

The first run installs torch and transformers (a few minutes). The CLIP model must be
in the Hugging Face cache; fetch it once with
`uv run --extra vision python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-base-patch32')"`.

## What you see

A certificate for CLIP's text codes over about 170 words, then a ten-word sentence
heard word by word and recalled in order.

## The idea

CLIP was trained on hundreds of millions of image-caption pairs; the brain was
trained on nothing. They meet through codes. The certificate's first line shows
something about CLIP that matters: its raw codes sit in a **cone** -- the mean
cosine between two unrelated words is about 0.8. Stored as they are, everything
would look like everything else. **Centring** (subtracting the vocabulary's mean
code) removes the cone, and the codes then behave almost like random white ones: the
"spectral overlap" near 1 says so. The certificate's columns compare CLIP's codes
with ideal white codes, load by load.

The `Sense` does the centring for you, fitted once on the vocabulary.

## Exercises

1. **Predict, then run.** From the certificate, how long a sentence can the brain
   hold at 90%? Write a sentence one word longer and one word shorter, and test.
2. Use only 20 words as the vocabulary. Does recall get easier or harder? Check the
   certificate's notes.
3. Replace `CLIP("text")` with `GaussianCodes(dim=512)` (from `ubrain.codecs`), with
   integers as items. Compare its certificate with CLIP's. Where does the cone go?
4. Store "cat" and "kitten" in one sentence. Do they get confused at recall? Look at
   their cosine (`sense([a]) @ sense([b]).T`) and explain.
5. **Going further:** write your own codec -- any class with `name`, `modality`,
   `dim`, `digest`, `license` and `encode(items)` -- for example, a bag-of-letters
   code of words. Certify it. Is it better or worse than CLIP for this memory, and why?
