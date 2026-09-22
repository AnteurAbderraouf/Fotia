# Flame in Freefall

Turning NASA's scattered microgravity combustion research into tables you can
actually model on.

**NASA Space Apps Challenge 2026** — *Flame in Freefall: AI-Powered Fire Safety
Insights from Microgravity Combustion Data.*

---

## The problem this repo solves

NASA has flown combustion experiments on the ISS, the Space Shuttle and
parabolic-flight aircraft for decades. The results are published, but they live
in 25 separate investigations with no common format: different spreadsheet
conventions per research team, outcomes buried in free text, headers that lost
their subscripts to encoding, Excel formula errors frozen into cells as if they
were measurements.

That fragmentation *is* the challenge. So the catalogue here is not preparation
for the real work — it is half the deliverable.

## What is in here

```
combustion_science/     25 NASA investigations, one folder each
  <PSI-xxx>/
    csv/                working copies of the usable tabular files
    info.md             NASA's own description, verbatim
    layout.json         how to unflatten NASA's bulk download (where needed)

flame/
  common/               shared cleaning, paths, and a standard table audit
  loaders/              one loader per investigation — the cleaning IS the code

data/processed/         the cleaned tables (regenerable, see below)
scripts/build_all.py    rebuilds every table from scratch
```

## Rebuild everything

```bash
pip install -r requirements.txt
python scripts/build_all.py
```

`combustion_science/` is read-only source of truth — nothing ever writes into
it. `data/processed/` is fully disposable: delete it, run the script, and it
comes back identical.

## The datasets, honestly

Ten investigations yielded usable tables. Row counts are what survived
inspection, not what the file appears to contain.

| Investigation | Rows | What it answers | Outcome type |
|---|---|---|---|
| PSI-159 ACME CFI-G | 272 | does a gas flame sustain itself or self-extinguish? | classification, 67% baseline |
| PSI-69 FLEX-1 | 213 | how much CO₂ or helium puts a droplet fire out? | classification, 87% baseline |
| PSI-39 CFI | 146 | does an invisible cool flame persist after the visible one dies? | classification, 85% baseline |
| PSI-101 SAME-R | 134 | what do the ISS smoke detectors actually see? | regression |
| PSI-25 BASS-II | 129 | 129 real ISS burns of spacecraft materials | descriptive only |
| PSI-142 Princeton | 95 | extinction limits, ozone sensitisation (**1 g**) | regression |
| PSI-107 SPICE | 70 | how much soot does this fuel make? | regression |
| PSI-117 FLEX | 141 | cool-flame diameters across alkanes | mixed measurement/simulation |
| PSI-115 | 35 grids | smoke plume with and without gravity | 5 simulated cases |
| PSI-99 Saffire-II | 9 | the same material burned in orbit and on Earth | comparison card |

Fourteen further investigations are catalogued but hold no per-test outcome —
condition lists, test matrices, per-day logs, image indexes. Knowing which
sources are dead ends is itself a result, so they stay in the catalogue.

### Things the data does not support

Stated plainly, because a fire-safety tool that oversells itself is worse than
none:

- **PSI-25 has no trainable label.** Only 20 of 129 tests state an outcome in
  the free-text field; the rest list an airflow ramp with no conclusion.
  Oxygen consumption looked promising but is physically impossible in places —
  final O₂ reaches 40.4% and the delta runs from −19.7 to +20.6 points, meaning
  oxygen *appears*. The chamber was repurged between readings. It remains an
  excellent materials catalogue.
- **PSI-101 is a regression, not an alarm classifier.** No column records a
  detector firing. Inventing a voltage cutoff would be inventing the label.
- **PSI-117 mixes measurements with simulations of the same conditions.** Rows
  carry a `source` column; any train/test split must be by physical condition,
  not by row, or the score is meaningless.
- **PSI-142 is ground data at 1 g.** Every table carries a `gravity` column so
  regimes are never pooled by accident.
- **PSI-115 has no physical axes.** Its coordinate meshes fail to parse, so
  plume widths are in grid cells, not millimetres. Temperatures are normalised
  ratios, not kelvins.

### Modelling constraints, decided up front

Row counts are in the hundreds, so this is classical machine learning:
logistic regression and shallow trees, not deep learning or large ensembles,
which would memorise noise at this sample size. Every loader separates
`FEATURES` (known *before* ignition) from post-burn measurements, in code
rather than by convention — predicting extinction from the extinction diameter
would score beautifully and mean nothing.

## What is deliberately not in this repository

The NASA archive itself: the unflattened `raw/` trees, the original download
archives, the PDF reports, and PSI-115's CFD grids — 4.8 GB in total. Git keeps
version history, it is not a backup tool, and GitHub rejects files above
100 MB. Everything tracked here comes to under 8 MB.

All of it is public and re-downloadable from
[psi.nasa.gov](https://psi.nasa.gov); `layout.json` and `scripts/unflatten.py`
reconstruct NASA's folder structure from the bulk-download archives, which are
flattened on the way out.

## Status

Data collection and cleaning are complete. Baseline models and the dashboard
are next. `handoff-2026-09-22.md` carries the full working record, including
eight claims an earlier pass got wrong that inspecting the data disproved.
