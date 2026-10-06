# camera_ready_checks (5 Oct 2026; Tier 2 additions marked)

Independent checks of the camera-ready draft (`paper_camera_ready/`), plus the analyses proposed for it.
The code is written from scratch: `common.py` and `c1`-`c3`, `c5`, `c6` import nothing from `scripts/` or `dif/`,
so agreement with the draft is a second implementation, not a re-run. `c4` calls `dif/anchored.py` on purpose,
because its question is what the published pipeline reports under a null.

## Inputs
- `data/responses_free25.jsonl` (main arm, 15,000 rows) and `data/responses_free25_temp0.jsonl` (5,000 rows).
  Both are in `data/` (RECOVERY.md expects them there).
- `data/items25_index.json` (item ids, stem length, answer type; no item text), `results/sapa_calibration.csv`
  (2PL a, b), `results/a6_matched_controls25.csv` (human controls, used by c5 and c6 only). c10 alone needs
  the item text, which is not distributed (see `data/README.md`).

## Run (numpy, scipy, pandas, matplotlib; about 1 h on 2 cores)
    python3 camera_ready_checks/c1_verify.py          # accounting, theta, lz*, item test, temp-0 single answers, sim null, one-item power
    python3 camera_ready_checks/c2_reductions.py      # item test under modal / single-run / run-mean reductions, three nulls
    NNULL=1500 python3 camera_ready_checks/c3_bundles.py   # letter-series and LN bundles: calibration, 2-D sensitivity, power
    python3 camera_ready_checks/c4_runlevel_nulls.py 0 100 &  python3 camera_ready_checks/c4_runlevel_nulls.py 100 200
    python3 camera_ready_checks/c4_runlevel_nulls.py merge     # run-level pipeline under three re-run nulls
    python3 camera_ready_checks/c7_humans.py          # SAPA: calibration check, real groups of eight, verbal-letter correlation (~12 min)
    python3 camera_ready_checks/c7b_ln_bundle.py      # LN subscale bundle at the SAPA verbal-LN correlation
    python3 camera_ready_checks/c8_purified.py        # Tier 1: item test with flagged items purged from the abilities (~30 s)
    python3 camera_ready_checks/c5_figures.py         # figures/*.pdf, *.png (EMPH_LETTER=0: the Tier 1 item figure, no item-type split)
    python3 camera_ready_checks/c9_item_properties.py # Tier 2: implied shift vs item a, b, stem length, answer type (~35 s)
    NREP=10000 python3 camera_ready_checks/c11_bundle_anchor.py   # Tier 2: letter-series bundle robustness grid (~25 min)
    EMPH_LETTER=1 python3 camera_ready_checks/c5_figures.py       # Tier 2 figures: letter series marked, VR.11 labelled
    python3 camera_ready_checks/c12_matched_humans.py # 7 Oct: real groups of eight matched to the models' abilities (~10 min)
    python3 camera_ready_checks/c11b_bundle_regression_null.py   # 7 Oct: letter-series bundle, null regressing to the human mean
    python3 camera_ready_checks/c13_macros.py         # 7 Oct: tables/ck_extra.tex (macros for c1 power, c11/c11b, c12)
    NNULL=1500 python3 camera_ready_checks/c6_tables.py    # tables/*.tex and tables/ck_numbers.tex (macros, prefix \ck)
    ICAR_ITEMS=... python3 camera_ready_checks/c10_exposure.py   # NOT RUN: needs the item text and the infini-gram API
All seeds are fixed in the scripts. The paper takes its camera-ready numbers from `tables/ck_numbers.tex` (macros, prefix `\ck`).

## What the checks found
1. **Reproduced.** Accounting (600 runs = 8 x 5 x 15; 14,138 parsed; median 571 per item), the run-level rate 0.84 with
   the published pipeline, all eight model thetas, lz* (-0.97 to +0.49) and the modal item test (same five items,
   same p-values) all match the draft. From the raw SAPA file: N = 95,166, 732,720 responses, and an independent MML
   calibration recovers every a and b within 0.003. Real groups of eight flag 0.0065 of items on average (draft: 0.006).
2. **Run-level nulls.** With eight invariant synthetic humans re-run like the models, the pipeline flags
   0.75 (90% range 0.56-0.92) when one answer is copied and flipped (the draft's null), 0.71 (90% range 0.48-0.92) when runs share fixed propensities
   matched to each model's run-to-run variance, and 0.00 when runs are independent. The inflation comes from the
   dependence among runs, not from the flips.
3. **Reductions.** The model-level rate is 0.20 with modal answers, 0.12 on average with one random run, and
   0.08 with the run mean under the stop-loss bound (valid whatever the dependence among runs). VR.36 and VR.39
   are flagged by all three in both arms. Under a stochastic-subject null the modal reduction alone flags 0.059.
   Against 1,000 real groups of eight tested the same way: p <= 0.001 (modal), 0.003 (single run), 0.013 (run mean;
   0.003 when each person's ability rests on at least eight other items).
4. **Power.** One item at a time, median power is 0.06 / 0.48 / 0.82 / 0.97 at 1 / 2 / 3 / 4 logits when the item is
   easier for models, and about 0 when it is harder.
5. **Letter series (exploratory, post hoc).** Run mean 6.0 correct over 56 model-item pairs against 15.7 expected from
   verbal ability. c3 and c7 first calibrated it at p 0.017 / <0.001 / 0.001 (fixed / fresh / matched nulls) and 0.012
   at the SAPA latent correlation of verbal with letter-series ability, 0.73 (95% CI 0.72-0.74; with number series
   0.88). The Tier 2 grid (c11, item 8) supersedes these single values: from below 0.001 to about 0.11, depending on
   run dependence, dimensionality and the verbal anchor. The LN subscale (all nine items) is weaker at its own
   correlation, 0.79: p = 0.14 (c11; c7b gave 0.15).
6. **Draft text.** "Scores vary less than binomial sampling predicts" (SD 0.0835 vs 0.1014) ignores item
   heterogeneity: a Poisson-binomial with the observed item proportions predicts 0.081.

7. **Purified abilities (Tier 1, c8).** Re-estimating each model's ability without the flagged items, iterated to a
   fixed point, keeps VR.36 flagged in all 12 combinations of reduction, arm and ability estimate and VR.39 in 11.
   The exception, main-arm modal answers, is degenerate: 9 items are removed and 4 models answer none of the 16 left,
   so their abilities sit at the -4 bound. Reproduced with the author's own estimator (scripts/a12_eight_examinees25.py).

Since Tier 1, `tab_runlevel_v2.tex` and `fig_runlevel` take the copied-answer-plus-flips null from the paper's own
`results/a13_replicated_human_null25.csv` (0.75, 90% range 0.52-0.92), so each null has one source in the paper;
the c4 re-implementation of that null (0.75, 0.56-0.92) stays here as a cross-check. Macros now round half up.

8. **Tier 2 (c9, c11; c10 not run).**
   - *Bundle robustness (c11).* The letter-series result depends on three choices, all varied: run dependence
     (fixed / fresh / matched), dimensionality (1-D, or letter ability departing from verbal ability as much as two
     human abilities correlated 0.73), and the verbal anchor (all 16 VR items, without VR.36 and VR.39, or without all
     four flagged verbal items). Calibrated p runs from below 0.001 to about 0.11 across that grid (see
     `results/c11_bundle_anchor.json` and `tables/tab_bundle.tex`), and none of it allows for the bundle having been
     chosen after inspecting Table IX. The paper now calls the result post hoc and gives the range, not one p-value.
     An independent check also noted: (a) the 2-D null is centred on verbal ability; regressing towards the human mean
     would make a two-sided p about 0.08, because excesses then dominate the null (one-sided, the deficit becomes
     rarer still); (b) rho-hat comes from per-domain calibrations whose letter slopes are about 1.2 times the 25-item
     ones; generating the null on those scales gives 0.019 instead of about 0.012. c7's `letter_bundle_at_rho` (2,000
     data sets) is superseded by c11; the two agree within Monte Carlo error.
   - *Item properties (c9).* The implied difficulty shift (the logit change that reproduces the models' run-mean
     total; abilities from each model's other items; capped at +-6) shows no clear relation to human a or b (Spearman
     -0.14 and 0.09), and relates to stem length (-0.40) only through the three short antonym stems (-0.15 without
     them). The three antonym items (VR.11, VR.26, VR.36) are all easier for the models than predicted, six of the
     seven letter series harder: a post hoc pattern by item type, reported as such.
   - *Exposure (c10).* Written and tested offline with a mocked API, not run: api.infini-gram.io is blocked from
     this environment and from the linked computer's sandbox. On any machine with internet access, run
     `python3 camera_ready_checks/c10_exposure.py` (about 250 requests, a few minutes); `--dry-run` prints the
     queries. It counts each stem (and each series, and the odd-one-out option sets) in the OLMo-2 training data and
     four open corpora, and correlates exposure with the c9 shift (memorisation predicts a positive correlation),
     pooled and for the two OLMo-2 models against their own training data.

## Files
- `results/` JSON and CSV outputs of c1-c4 (c4 in parts: `c4_part_000.csv` reps 0-39, `c4_part_040.csv` 40-119,
  `c4_part_120.csv` 120-199; merged in `c4_runlevel_nulls.csv`).
- `figures/` fig_items (Fig. 4), fig_runlevel (Fig. 1), fig_letter (Fig. 5).
- `tables/` tab_reductions, tab_bundle, tab_runlevel_v2, tab_power_v2, ck_numbers.
- `ck_refs.bib` seven references to append to `refs.bib` (DOIs checked on 5 Oct 2026).

## 7 October 2026 (v1.1.0)
- **Item text removed.** The ICAR item text and keys, and the model output text, option orders and key
  positions in the response files, are no longer distributed (PsychArchives Scientific Use Licence; see
  `DATA_LICENSES.md`). The checks read scored outcomes only and give the same results.
- **c12, matched human groups.** c7's groups of eight average theta +0.23, while the models sit at -0.93 to
  -2.08, and a person's ability rests on about seven items (median) against a model's 24. c12 repeats the
  calibration with each group matched to the models' abilities (each model contributes one person whose
  ability, estimated from his or her other items, lies within 0.25 logits of the model's).
  - Mean flag rate: 0.011 (exact test) and 0.006 (stop-loss), against 0.007 and 0.005 unmatched.
  - p for the observed rates: 0.001 (modal 0.20), 0.015 (single run 0.12) and 0.032 (run mean 0.08),
    against <= 0.001, 0.003 and 0.013 unmatched.
  - No group reaches 0.20; the largest flags 0.16.
  - With at least eight other items per person: 0.001, 0.007 and 0.011.
- **Bundle Type-I with estimated abilities.** The stop-loss bound is valid when abilities are known. With
  abilities estimated, as in c11's null data sets, the uncalibrated run-mean bundle test at a nominal 0.05
  rejects in these shares of null data sets (`typeI_005` in `results/c11_bundle_anchor.json`):
  - fixed runs: 0.08 (1-D) and 0.16 (2-D);
  - matched runs: 0.08 (2-D) and 0.006 (1-D);
  - fresh runs: at most 0.04.
  That is why every reported p is calibrated against simulated nulls.
- **Bundle null regressing to the human mean (c11b).** If letter-series ability given verbal ability regresses
  towards the human mean, as in people, most null data sets show an excess. The calibrated two-sided p
  (4,000 null data sets) is then:
  - 0.08 with matched runs;
  - 0.12 with fixed runs;
  - 0.04 with fresh runs.

  One-sided, the observed deficit stays rare: p <= 0.004 under all three.
