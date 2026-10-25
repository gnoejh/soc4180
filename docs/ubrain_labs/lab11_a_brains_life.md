# Lab 11 -- A brain's life: save, load, datasheet and monitor

**Goal:** treat a brain as a device: save its knowledge, restore it exactly, read its
datasheet, and watch it run.

## Run

```bash
uv run examples/11_save_and_load.py         # a second
uv run examples/12_monitor_and_datasheet.py # a few seconds
```

## What you see

**11:** a brain learns a bandit for 3000 trials and is saved (a file under a
kilobyte). A brain loaded from the file and the one that never stopped then see the
same 3000 trials and choose the same action every time. Loading against another body,
or with another codec, is refused with the reason. **12:** the `holder` brain's
datasheet row (items recalled of a list of 16: 3.93 at d = 128), a measurement of it,
the health of an episode at and past its capacity, and a second brain that watches
the first through its monitor channels.

## The idea

**A brain's state is its knowledge.** An untrained brain learns all its life, so
saving it is saving what it learned. `brain.save(path)` writes every region's state
with a header naming the wiring, the body and every codec; `Brain.load(path,
manifest)` rebuilds it, and refuses a body or a codec that is not the one it learned
with -- a brain that learned one body's channels would misread another's.

**A datasheet, like a chip's.** Every figure on it is a closed-form law for one
region, with the conditions it holds under (`ubrain.datasheet`). `health(brain)`
compares what a region holds now with its law; past its capacity, the row says so.

**A brain watching a brain.** `monitor_manifest(brain)` turns each scalar a region
reports into a channel, so the state of one brain can be the senses of another.

## Exercises

1. **Predict, then run.** Set the buffer's decay to 0.9
   (`w.regions["prefrontal.buffer"]["gamma_m"] = 0.9`) and print the datasheet again.
   Predict the new figure with `buffer_count` first.
2. In example 11, the loaded brain and the original choose identically for 3000
   trials. What would break that if the loaded brain had its own random generator?
3. Why must loading refuse a codec that was refitted, even when it has the same name?
4. The health check measured the episode a little above its law at its capacity.
   Lab 7 met the same effect. What do the two settings share?
5. **Going further:** write a watcher brain that raises a flag when the holder's
   `dropped` channel rises. Which profile's regions could act on that flag?
