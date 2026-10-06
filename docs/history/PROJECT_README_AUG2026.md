> **Historical document (August 2026), lightly edited in October 2026.** Kept for reference on how the
> machine administration was run.
> It describes the earlier 8-item phase and file names that have since changed; its numbers are not
> the paper's. ICAR item text is no longer distributed with this repository (see `DATA_LICENSES.md`
> and `data/README.md`). For the current analyses, see the top-level `README.md` and `RECOVERY.md`.

# Measurement Invariance Between Human and Machine Respondents

Research artifact for the paper *"Measurement Invariance Between Human and Machine
Respondents on Cognitive Assessments"* (anonymous submission).

Everything here runs on **two CPU cores**. No accelerator is used at any point in the
analysis. The only step that benefits from a GPU is administering items to models,
which is done once on free-tier Colab and exported as a response matrix.

---

## What this is

Claims that a model reaches "human-level" performance on a cognitive test compare two
populations on one scale. That comparison is interpretable only under **measurement
invariance**. This repository tests that assumption.

The direct test is unavailable: calibrating item response models on a pool of machine
respondents lands in the small-$N$/large-$J$ regime where such models are documented
to fail. We use an **anchored** design instead — item parameters come from a large
human sample, and the machine group contributes only a two-parameter latent
distribution plus one studied item at a time.

---

## Layout

```
dif/                     estimation library (numpy + scipy only)
  irt.py                 2PL model, MML-EM calibration, EAP scoring, group distribution
  anchored.py            anchored LR-DIF, purification, Raju areas, BH control
  mh.py                  Mantel-Haenszel DIF, ETS A/B/C classification
harness/                 machine administration (the only GPU-adjacent part)
  administer.py          prompts, parser, option-order randomisation, admin loop
  items.py               item-bank validation + procedural letter-series generator
  Colab_Administration.ipynb
scripts/
  prep_data.py           download-free loading + validation of human response data
  run_sims.py            validation studies A (separation), B (focal N), C (null)
  run_human_controls.py  placebo floor + human-subgroup DIF calibration
  run_invariance.py      THE primary analysis (human vs machine)
  make_tables.py         results -> LaTeX fragments (no hand-typed numbers)
paper/                   IEEEtran source, bibliography, generated tables
PREREGISTRATION.md       hypotheses, protocol, stopping rules — registered in advance
```

## Running the remaining step

See **`RUNBOOK.md`** for precise step-by-step instructions for the Colab
administration — the one part that cannot run in a sandbox without network access
to HuggingFace and icar-project.org.

## Reproducing

```bash
make data       # load + validate human response matrices
make sims       # validation studies (~25 min on 2 cores)
make controls   # placebo + subgroup calibration (~40 min on 2 cores)
make tables     # regenerate all LaTeX fragments from results/
make paper      # compile paper/main.pdf
```

Then, once the Colab step has produced `machine_responses.parquet`:

```bash
python3 scripts/run_invariance.py --machine machine_responses.parquet \
                                  --meta machine_meta.csv
make tables paper
```

---

## Data provenance

| Dataset | What | N | Source | Licence |
|---|---|---|---|---|
| `ability` / `iqitems` | ICAR-16 cognitive ability, scored and raw option choices | 1,525 | `psychTools` R package (CRAN mirror) | GPL (≥2); ICAR items are public domain |
| `bfi` | Big Five Inventory + gender/education/age | 2,800 | `psychTools` | GPL (≥2) |
| `spi` | SAPA Personality Inventory + demographics | 4,000 | `psychTools` | GPL (≥2) |
| H-ARC | human ARC solutions + action traces | 1,729 | OSF `bh8yq` | see repository |

`scripts/prep_data.py` reads the `.rda` files directly with `pyreadr`; **no R
installation is required**.

### Item text — included and verified

The 8 text-administrable ICAR items are transcribed in
`data/derived/icar_items_text.json`, taken from the archived PsychArchives / ZPID
release of the ICAR item set. (The ICAR project website is not a usable route: its
registration path directs to a lapsed domain and the item pages are password-gated.)

`scripts/verify_items.py` gates administration on two independent checks: every
transcribed key matches the published scoring key shipped with the response data
(key list omitted here), and the deterministic series items are re-solved from
their stems. It exits non-zero on any mismatch.

**Scope condition (registered).** Only the 8 verbal-reasoning and letter/number-series
items are administrable to a text-only model; matrix-reasoning and rotation items are
figural. The text-only subset is the pre-registered primary analysis set. A
vision-capable arm can use all 16 via `--all-items`.

**H-ARC.** Download `osfstorage-archive.zip` from OSF and extract into `data/raw/`.

---

## Status

| Component | State |
|---|---|
| Estimation library | complete; 2PL recovery r = 0.99, DIF power 1.00 / FP 0.06 on synthetic |
| Study A — operating envelope | complete, 360 replicates |
| Study B — focal sample size | complete, 280 replicates |
| Study C — null calibration | complete, 300 replicates |
| Study D — anchor uncertainty | complete, 400 replicates (limitation found and diagnosed) |
| ICAR-16 calibration | complete, real data N = 1,525 |
| Random-split placebo | complete, 320 splits on real data |
| Human-subgroup baseline | complete, matched focal N = 300 |
| Administration harness | complete, unit-tested (parser, renderer, key validation) |
| Tests | **4/4 PASS** — `tests/run_all_tests.sh` (item provenance, reproducibility, administration loop, end-to-end) |
| Paper | 8 pages, IEEEtran, builds clean, 30 references, 2 figures, 6 tables |
| ICAR item text | complete, verified against the published key |
| **Machine administration** | **pending — requires the Colab run** |
| **Human-vs-machine result** | **pending — blocked on the above** |

See `results/SUMMARY.md` for the headline numbers.

### Runtime note

Set `OMP_NUM_THREADS=1` (and the OpenBLAS/MKL equivalents) before running the
multiprocessing scripts. With 2 cores and 2 worker processes, multi-threaded BLAS
oversubscribes and costs roughly 2.5x wall clock. `run_all.sh` does this for you.

The paper is not submission-ready until the machine administration has been run. Every
other component is finished, and the Results section is produced by
`run_invariance.py` and `make_tables.py` without further analytic choices.
