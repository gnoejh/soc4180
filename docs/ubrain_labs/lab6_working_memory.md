# Lab 6 -- Holding a list: working memory in the prefrontal buffer

**Goal:** see a list held by position in one decaying vector, predict how many of its
items survive, and see why rehearsal does not buy more.

## Run

```bash
uv run examples/06_working_memory.py
```

## What you see

The `holder` brain hears a list of 16 items and is asked for each position. The
table gives, per position, its age (writes since it was stored), its weight, the
law's recall rate and the measured one, before and after a rehearsal cycle. The
newest positions are always recalled; the oldest almost never. At the profile's
settings (d = 512, V = 256, decay 0.75 per write) the law says 6.63 items are held
on average, and 1000 brains hold 6.67.

## The idea

**One vector, many items.** The buffer binds item i to position i's role and adds it
into one vector. Every write first multiplies what is there by gamma (0.75), so an
item written a steps ago carries weight gamma^a. Reading a position unbinds its role
and picks the nearest item the sense knows.

**Why recall fails.** The other items are noise in that read. An item is recalled
when its weight stands clear of the crosstalk of all the others, so recall falls
with age, and how fast depends on d (more dimensions, less crosstalk) and V (more
items to confuse it with). `ubrain.datasheet.buffer_count(L, gamma, d, V)` sums the
recall rate over the L positions: the expected number of items held.

**Why rehearsal does not help here.** A full rehearsal cycle rewrites every position
once, which multiplies every term by the same factor. The read compares an item with
its competitors, and a common factor cancels: recall comes back identical, position
by position.

## Exercises

1. **Predict, then run.** Before changing anything, compute
   `buffer_count(16, 0.9, 512, 256)` in Python (`from ubrain.datasheet import
   buffer_count`). Then set the decay to 0.9 in the example (`buf["gamma_m"] = 0.9`
   before it is read) and run. Does the measurement follow your prediction?
2. Double `D` instead. Which buys more items held at decay 0.75: doubling `D`, or
   moving the decay from 0.75 to 0.9? Predict both with `buffer_count` first.
3. The newest four positions are recalled almost perfectly at every setting you have
   tried. Using the weights column, explain why the oldest items fail before the
   newest ones feel any crosstalk.
4. Rehearsal left recall unchanged. Design a rehearsal schedule that *would* change
   it (hint: rehearse only some positions), and say which positions it would help and
   which it would hurt.
5. **Going further:** the classic span of human working memory is "seven, plus or
   minus two". Find settings of decay and `D` where the buffer holds about seven.
   Is the answer unique?
