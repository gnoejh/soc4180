# Lab 7 -- From hippocampus to cortex: consolidation

**Goal:** see a fast one-shot store hand its contents to a slow store, and predict
what survives each stage.

## Run

```bash
uv run examples/07_consolidation.py      # under a minute
```

## What you see

The `rememberer` brain sees episodes of K = 4 items. At each episode's end the
hippocampus retrieves its items and replays them into the cortex under fresh keys.
The table has one row per number N of pairs the cortex holds, with three rates: the
hippocampus's retrieval, the cortex's faithful recall of what it was given, and the
end-to-end rate, each beside its law. At d = 128 one episode holds about 5 items at
90%, while the cortex holds about 569 pairs.

## The idea

**Two stores, two laws.** The episode is a vector store: an item of a k-item episode
is read at the decode law's rate in dimension d (`ubrain.observe.decode_law`). The
cortex is a matrix store, W += z k^T, read by z = W k: its crosstalk is a product of
two overlaps, so it behaves like a vector store in d^2 dimensions
(`ubrain.datasheet.weighted_law(1, N, d*d, V, d_self=d)`). That is why the cortex
holds hundreds of items where an episode holds a handful.

**The product.** A memory recalled from the cortex needs both stages right, so the
end-to-end rate is the product of the two. The cortex stores what the hippocampus
retrieved, errors included.

**Two cautions the example states.** The laws are first-order: at a dimension as
small as 128, measurements land at or a little above them near the cortex's
transition. And the cortex's law counts N *different* pairs: every item in this run
is distinct.

## Exercises

1. **Predict, then run.** Set `K = 8`. Before running, compute the hippocampus's rate
   with `decode_law(8, 128, 4000)`. Which stage now limits the end-to-end rate at small
   N, and which at large N?
2. Set `D = 256`. The episode's capacity grows in proportion to d, the cortex's with
   d squared. Predict the new cortex capacity with `ubrain.datasheet.matrix_capacity`
   (it is past the table's largest N, so add larger counts to `EPISODES`) and check it
   against the row where recall crosses 90%.
3. **Break the assumption.** Make items repeat across episodes (draw them from a
   vocabulary of 500 instead of giving each a fresh one). The cortex's recall falls
   below its law once N passes the vocabulary. Explain why copies of one item add
   crosstalk together.
4. Why does the hippocampus's rate stay flat down the table while the cortex's falls?
5. **Going further:** a brain that replays each episode twice writes every item twice
   under the same key. What does that do to the item's weight, and to the crosstalk
   the others feel?
