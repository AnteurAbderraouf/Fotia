# Project Handoff — Flame in Freefall (NASA Space Apps 2026)

Read this fully before doing anything. This summarizes everything decided and
built so far in a separate chat session (claude.ai), so you don't have to
rediscover it. The user (Raouf) wants to continue the project with you
directly on his filesystem from here on.

## The challenge

NASA Space Apps Challenge 2026, "Flame in Freefall: AI-Powered Fire Safety
Insights from Microgravity Combustion Data." Goal: build an interactive,
AI-powered dashboard that summarizes, ranks, and interprets NASA's scattered
microgravity combustion research (mostly from the ISS Combustion Integrated
Rack) — the data exists across dozens of separate investigations with no
common format, and that fragmentation is itself the problem the challenge
wants solved.

Timeline: several weeks, not a rushed few days. The user has some CV/image
processing exposure but hasn't done AI/ML work before — treat this as a
learning project, not just an execution task. He wants to understand
mechanisms, not just get code that works.

## Where the data lives

**NASA Physical Sciences Informatics (PSI)**, psi.nasa.gov — the actual data
repository, separate from general NASA description pages (which have no
downloadable data, just narrative). PSI organizes data into investigations,
each with an ID like PSI-117. Investigation pages are a JS single-page app,
not crawlable/fetchable by a simple HTTP request — the user browses them
manually in his own browser and downloads files himself.

**Critical distinction NASA uses:** every investigation's "Project Type" is
either:
- **Flight Investigation** — real ISS/CIR experiments, has actual raw +
  analyzed data (this is what you want)
- **Ground Investigation** — CFD/modeling studies that usually just *cite*
  flight data from another investigation for validation; typically has no
  usable dataset of its own, just PDFs/reports

Check this before spending time opening an investigation's file tree.

**Investigation folder structure varies a lot between investigations** —
no standard layout across PSI. Examples seen so far:
- PSI-117: `Analyzed_Data/`, `Presentations/`, `Reports/`, `metadata/`
- PSI-159: `Investigation Metadata Files/`, `Reports/`, `Engineering
  Documents/`, `Experimental table/`, `Raw Data/`, `Representative Image(s)/`

**"Raw Data" folders are dangerous to grab wholesale** — one investigation's
full Raw Data was 1.9 TB (video/camera output across years of testing). Only
ever download a single specific test run's files, never select-all on a Raw
Data folder.

## Local folder structure (already set up by the user)

Location: `C:\Users\anteu\Documents\Nasa\combustion_science\microgravity_investigation\`

One folder per PSI ID, directly in that root (not nested under a `data/`
folder), each containing:
```
PSI-XXX/
  csv/        <- flattened, working copies of usable numeric data
  reports/    <- working copies of useful PDFs (papers, presentations)
  info.md     <- structured metadata + notes (see template below)
```

Recommended refinement (not yet fully applied — up to you/user which
investigations get this): also keep a `raw/` subfolder mirroring NASA's
original folder names/structure exactly as downloaded, so nothing is lost
even after you've picked out what's useful into `csv/`/`reports/`.

The user has scaffolded (empty, TBD) folders for ~19 more investigation IDs
already: PSI-159, PSI-10, PSI-101, PSI-98, PSI-39, PSI-102, PSI-106, PSI-99,
PSI-107, PSI-100, PSI-21, PSI-25, PSI-20, PSI-68, PSI-22, PSI-23, PSI-69,
PSI-26, PSI-47. These need their `info.md` filled in and csv/reports
populated as the user finds and shares their investigation description
pages and downloadable data.

### info.md template

```markdown
# PSI-XXX

## Project

**Project Type:** [Flight Investigation | Ground Investigation]
**Proposal Title:** ...
**Investigation Start Date:** ...
**Investigation End Date:** ...
**Sponsoring Agency:** ...
**NASA Center:** ...

## Objectives
[full text as given by NASA, not summarized — the user wants exact reproduction]

## Approach
[full text]

## Hypothesis
[full text]

## Research Impacts/Earth Benefits
[full text]

## Contact(s)
[names]
```

IMPORTANT: when the user pastes an investigation's NASA description text and
asks you to "md" it, reproduce the FULL text verbatim in markdown — do not
summarize or condense it. He explicitly corrected this behavior once
already. Summarize only in a separate "why it's here / how it relates to
other investigations" section you add below the verbatim NASA text, when
useful (e.g., to note whether it's a dead-end for data purposes, or connects
to another investigation).

## Investigations catalogued so far (findings)

- **PSI-117** (Flight Investigation) — THE main data source found so far.
  Cool-flame droplet combustion (NASA FLEX experiment), fuels n-heptane/
  n-decane/n-dodecane, ISS Combustion Integrated Rack. Has a genuine
  `Analyzed_Data` folder with real per-condition scalar CSVs (pressure,
  droplet diameter, O2/N2/He fractions -> extinction diameter, burning rate
  K). These CSVs are messy multi-block exports (see "CSV cleaning" section
  below) but usable. ~93 pooled rows extracted so far (see master table
  section). Key papers: Farouk & Dryer 2014, Farouk et al. 2019, Farouk &
  Dryer 2023 (see PSI-117 presentations for full reference list).

- **PSI-159** — Experimental table folder likely has usable per-condition
  data (structure: Investigation Metadata Files / Reports / Engineering
  Documents / Experimental table / Raw Data / Representative Image(s)).
  NOT YET actually opened/verified by the user — check "Experimental table"
  first, "Raw Data" only if that's empty and you have no better option.

- **PSI-62** (Ground Investigation) — CFD modeling only, validates against
  PSI-117's FLEX data, no new experimental data of its own (its "data"
  folder was a 9 KB metadata-only zip). NOTE: there was earlier confusion
  where a ground-investigation metadata zip named
  "PSI-117_metadata_NEWID-13_Effect_of_External_T.zip" was initially
  mis-attributed to PSI-62 — it actually belongs filed under PSI-117 (see
  that investigation's info.md, "Associated ground-investigation metadata"
  section). Don't re-confuse these two.

- **PSI-142** (Ground Investigation) — ozone-sensitized counterflow flame
  studies validating FLEX cool-flame flammability limits. No indication yet
  of a numeric dataset; likely another citation/context-only entry. Full
  NASA text captured in its info.md.

- **PSI-115** (Ground Investigation) — pyrolysis smoke growth simulation
  (dibutyl-phthalate particles), gravity vs. microgravity comparison. This
  is a DIFFERENT combustion regime (smoke/particle transport, not flame
  extinction) — don't pool its data into the PSI-117 master table, treat as
  a separate feature set. Has real raw simulation output: `.dat` files,
  flattened 2D CFD grid dumps (960x240 cells) for Gravity and NoGravity
  cases at 8cm and 10cm sample sizes. Variables per case: Temperature, U/V
  velocity (+ uR/vR variants), NumDen/NumDenR (number density), PSD/NS_PSD
  (particle size distribution, 1D not 2D), Smoke, SpeciesMF (species mass
  fraction), Vort (vorticity), DAM, NucRate (nucleation rate), plus a
  `grid3d.dat` coordinate mesh file and `nodeDiameters.dat`. Already
  converted the full "8cmGravity*" set to clean CSVs (see Scripts section) —
  NoGravity counterpart parts likely still need the same treatment. This is
  earmarked as a "Phase 3" stretch dataset for spatial/CV feature
  extraction, not immediate tabular modeling.

- **PSI-60** (Ground Investigation) — soot formation in 1g vs 0g flames,
  Smoke Point in Coflow Experiment (SPICE) data, DSLR image analysis +
  color-ratio pyrometry. Full NASA text captured in its info.md. Not yet
  checked for actual downloadable numeric data.

## Modeling plan (agreed, not yet executed beyond the PSI-117 master table)

**Why classical ML, not deep learning:** dataset sizes are small (tens to
low hundreds of rows per investigation, ~93 pooled so far from PSI-117
alone). Rule of thumb used: 10-50 rows per feature for classical ML
(RandomForest/gradient boosting), meaning realistically aim for ~50-100
pooled rows with only 2-3 features, and prefer simple/interpretable models
(logistic regression or shallow decision tree, max_depth~3) over complex
ensembles at this sample size — an ensemble will just memorize noise.
Report this as a stated methodological choice / limitation in the eventual
pitch, ideally with a learning-curve plot (accuracy vs. training set size)
to show statistical awareness rather than hiding the small-N problem.

**Risk label:** not present in the data — must be defined by the user, not
inferred. Simplest starting option agreed: binary "did the droplet sustain
cool-flame combustion past hot-flame extinction?" (cool flames are nearly
invisible/hard to detect, so persistence = elevated undetected-fire risk,
directly relevant to spacecraft fire safety). Other options discussed:
continuous extinction-diameter-based score, or a composite formula — pick
one and justify it, don't let this decision get skipped.

**Pipeline layers, build in this order, do not skip ahead:**
1. **Layer 1 (CSV, in progress):** clean per-investigation CSVs into ONE
   pooled master table with consistent schema:
   `investigation | fuel | source (expt/model/sim) | do_mm | pressure_atm |
   dext_hf | dext_cf | kavg | [other cols as they appear, missing values
   expected/fine]`. IMPORTANT caveat already identified: `expt` vs
   `model`/`sim` rows for the same physical condition are NOT independent
   data points — when doing train/test split later, split by physical
   condition, not by row, or you'll leak information.
2. **Train the baseline model** (RandomForest or simpler) on Layer 1 once
   pooling is "done enough" (~50-100 rows, don't over-invest before this).
3. **Layer 2 dashboard**, two distinct sub-parts, build in this order:
   - Streamlit app (fastest path to a working demo) — sliders for
     pressure/O2/fuel, model prediction on submit, show nearby real
     historical data points for comparison. `st.slider`, `st.selectbox`,
     `model.predict_proba`, this is intentionally simple.
   - Layer 2B, ONLY after the model works: a retrieval/grounding layer
     that surfaces relevant sentences from the PDFs/papers to explain WHY
     a prediction looks the way it does (physically grounded language,
     not just a bare number). This is what makes it "AI-powered" in a
     second distinct sense beyond the prediction itself.
4. **Phase 3 (stretch, only if time remains after 1-3 work):** raw
   video/image CV pipeline — extract droplet diameter from backlit video
   frames (threshold -> contour -> ellipse fit -> pixel-to-mm calibration
   -> plot `(d/d0)^2 vs t`, compare against known d-squared-law burning
   curves as a validation check), and/or the PSI-115 raw CFD field data
   (spatial feature extraction: max temperature, wake width, etc. from the
   960x240 grids). Do NOT start this before Phase 1-2 produce a complete
   working baseline — it's a genuinely bigger, riskier task (re-deriving
   measurements the original papers already computed) and easy to sink
   unbounded time into.

## Scripts already written and tested (should exist in the project or be
recreated — check /mnt/user-data/outputs equivalents from the prior session
if the user re-shares them, otherwise rewrite from the specs below)

### `convert_dat.py`
Converts raw NASA PSI `.dat` field-dump files (Fortran-style flattened grid
outputs, format VARIES between files — some are one-value-per-line, some
pack many values per line) into readable CSVs. Logic:
- Tries to parse a 4-int header line (`nx ny nz nvars`) on line 1.
- If found and nz==nvars==1: reads remaining whitespace/newline-separated
  floats with `np.loadtxt(path, skiprows=1)`, trims any trailing stray
  values to exactly `nx*ny`, reshapes to `(ny, nx)`, saves as a headerless
  grid CSV.
- If no numeric header: checks whether line 1 is a text label (e.g.
  "Diameter (starting from first non-vapor node)") — if so, skips it and
  reads the rest as a flat 1D array.
- Otherwise (no header, first line IS numeric): reads the whole file as a
  flat 1D array — this is correct/expected for particle-size-distribution
  (PSD) files, which are genuinely 1D, not a parsing failure.
- Ragged/irregular files (seen: `*_sGrid_grid3d.dat`, a coordinate mesh
  dump with inconsistent row widths) currently FAIL and are left alone —
  known unhandled edge case, only worth fixing if spatial coordinates are
  actually needed later.
- Writes a `manifest.csv` alongside the output folder logging status
  (`reshaped_2d` / `flat_no_header` / `flat_text_header` / `flat_fallback`
  / `FAILED`) per file, plus shape and notes — always check this after
  running, don't assume silent success.
- Usage: `python convert_dat.py /path/to/folder` (defaults to cwd), writes
  into a `converted/` subfolder.

### `clean_dat_csv.py`
Post-processing pass on the converted CSVs. Currently only fix implemented:
replaces `inf`/`-inf` with `NaN` (found in a `NucRate` nucleation-rate
field — a log-scale field where zero cells became `log(0) = -inf`).
Explicitly does NOT fill NaN with 0 — that would misrepresent "undefined"
as "zero nucleation," a real physical distinction the user needs to decide
on deliberately later, not have silently erased. Reports before/after
inf and NaN counts per file. Usage:
`python clean_dat_csv.py /path/to/folder` (defaults to cwd), writes into a
`clean/` subfolder, same filenames.

Both scripts handle two CSV shapes produced by `convert_dat.py`: flat files
have a single `"value"` column header; grid files are headerless numeric
matrices. Detect which by checking `list(df.columns) == ["value"]`.

### PSI-117 master table cleaning (ad hoc so far, not yet a reusable script)
Extracted from 4 "scalar per-condition" CSVs in PSI-117's Analyzed_Data
folder (MST_2024_Dext.csv, MST_2024_Kavg.csv,
PROCI_2023_Alkane_Extinction_Diameter_1atm.csv,
PROCI_2023_C12_Extinction_Diameter_vs_pressure.csv). These CSVs are messy
Excel/Origin exports: multiple experiment/model/fuel blocks laid out
side-by-side in the same file with pandas auto-generating `ColumnName`,
`ColumnName.1`, `ColumnName.2` etc. suffixes for repeated headers. Cleaning
approach: manually identify each block's real column triplet (e.g.
`Do, Dext_HF, Dext_CF` for one fuel/source combo, `Do.1, Dext_HF.1,
Dext_CF.1` for the next), `dropna` on the block's key column to isolate
just that block's real rows, tag with fuel/source/investigation metadata,
append to a list of dicts, then `pd.DataFrame(rows)`. Result so far: 93
rows spanning fuels C7/C10/C12 (heavily lopsided toward C12 — 69/93 rows —
flag this imbalance when reporting model results later). Two OTHER PSI-117
CSVs (`Burning_history_Flame_diameter.csv`, `Radiance_Flame_Diameter.csv`)
are TIME-SERIES curves (droplet diameter/radiance vs. time within one burn)
— structurally different, not yet incorporated into the scalar master
table; earmarked for later feature engineering (e.g., extract peak value,
burn duration as new scalar columns) rather than being pooled as-is.

This should become a proper reusable script (e.g. `build_master_table.py`)
that takes a `csv/` folder + a small per-file "recipe" (which column-blocks
map to which fuel/source) and outputs one pooled CSV — currently it's just
inline pandas code written fresh each time, which won't scale to the ~20
remaining investigations. Worth building this properly as one of your first
tasks.

## User's working style / preferences (apply throughout)

- Wants to understand underlying mechanisms before using abstractions;
  learns by building through doing.
- Asks pointed follow-up questions rather than accepting things at face
  value — expect to be pushed on claims, don't hand-wave.
- Wants direct, iterative work with minimal over-explanation once he
  understands a concept — but explain thoroughly the FIRST time something
  new comes up (he's new to AI/ML specifically).
- Corrects overreach when it happens (e.g. explicitly said "why don't you
  just md the whole text, why are you summarizing" when asked to convert
  structured NASA text to markdown) — when asked to reproduce/convert given
  text, reproduce it fully and exactly unless summarization is explicitly
  requested.
- Prefers being told the honest state of things (small dataset size,
  messy data, dead-end investigations) rather than optimistic framing.

## Immediate next steps (pick up here)

1. Verify PSI-159's "Experimental table" folder contents (user has not
   confirmed what's actually in it yet).
2. Build the reusable `build_master_table.py` script (see above) so adding
   each new investigation's CSVs to the pooled table doesn't require
   bespoke code every time.
3. Continue working through the ~19 scaffolded-but-empty PSI-XXX folders:
   for each, get the NASA description text (verbatim into info.md), check
   Project Type, check for a real Analyzed_Data/Experimental-table-style
   folder vs. Ground Investigation dead-end.
4. Once pooled master table reaches ~50-100 usable rows across
   investigations, stop data-gathering and build the baseline model +
   Streamlit dashboard (Layer 1-3 above) before returning to gather more
   data or attempting Phase 3 raw-data work.