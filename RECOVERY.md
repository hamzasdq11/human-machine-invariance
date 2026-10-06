# Recovery of the 25-item analysis (3 October 2026)

The August scripts that produced the accepted paper's 25-item results were lost
with the old workspace. The estimation library in `dif/` survived unchanged; the
data layer and analysis scripts below were rebuilt around it and checked against
every published number (gate 1). All match; see `results/`.

## Data (in `data/`)
- `sapaICARData18aug2010thru20may2013.csv`  (SAPA ICAR release, Harvard Dataverse doi:10.7910/DVN/AD9RVY)
- `responses_free25.jsonl`        main arm, temperature 0.7, 15,000 rows (scored outcomes; since v1.1.0
  without model output text, option orders or keys, see `DATA_LICENSES.md`)
- `responses_free25_temp0.jsonl`  pre-specified temperature-0 arm, 5,000 rows
- sensitivity variants: `python3 scripts/make_variants25.py` (unparsable as wrong, no frame v5); the
  lenient-parser variant is shipped (`*_lenient.jsonl`)

## Run (numpy, scipy, pandas; 2-4 CPU cores)
    python3 scripts/calibrate_sapa.py                                   # Table XI   -> results/sapa_calibration.csv
    python3 scripts/run_invariance25.py data/responses_free25.jsonl main      # Table X, 0.84
    python3 scripts/clustering25.py     data/responses_free25.jsonl main      # Table IX, variance shares
    python3 scripts/loo25.py            data/responses_free25.jsonl main      # leave-one-model-out
    python3 scripts/l2o25.py            data/responses_free25.jsonl main      # leave-two-out
    python3 scripts/anchor_path25.py    data/responses_free25.jsonl main      # purification path (A3)
    python3 scripts/human_controls25.py                                  # Table V (gender, placebo)
    python3 scripts/exact_cluster_bootstrap25.py data/responses_free25.jsonl main   # A5, ~2 h on 4 cores
Replace `responses_free25.jsonl main` with `responses_free25_temp0.jsonl temp0` for the temperature-0 arm.

## Gate 1 (published -> reproduced)
0.84 (21/25) -> 0.84 (21/25); same invariant items VR.16, VR.18, LN.03, LN.58;
mu -1.02 -> -1.017, sigma 0 -> 0; ETS B+C 0.96 -> 0.96 [withdrawn in the camera-ready, deviation D13:
Mantel-Haenszel matching on raw totals has no common scale here]; LOO 0.80-0.92 -> 0.80-0.92;
median deff 11.8 -> 11.8; variance shares 0.296/0.018/0.013 -> same;
gender 0.023/0.044 -> 0.017/0.049 and placebo 0.001/0.002 -> 0.001/0.002 (40 runs, new seeds).

## New
- Temperature-0 arm: 0.84 (21/25), sigma 0, mu -1.13.
- Exact cluster bootstrap (6,435 resamples): mean 0.86, 95% [0.72, 1.00]; leave-two-out 0.64-0.92.
- Purification: 4-anchor floor binds on pass 1 only; final set is a genuine fixed point.

## Required camera-ready analyses (run 3 October)
    python3 scripts/a1_model_fit25.py data/responses_free25.jsonl main   # A1 anchor-free model-level test (~20 min)
    python3 scripts/sim_observed25.py null 200                            # A4 null calibration
    python3 scripts/sim_observed25.py power 40                            # A9 simulated power
    python3 scripts/a6_matched_controls25.py                              # A6 matched controls, N = 600
    python3 scripts/a3_random_anchors25.py data/responses_free25.jsonl main   # A3 500 random anchor sets
    python3 scripts/a3_highdif_sim25.py                                   # A3 recovery at 80/88% DIF
Results: all 8 models reject (each p < 0.001; RMSD 0.29-0.35 vs human groups <= 0.077) [superseded the
same evening: under the replicated-respondent null below, that RMSD is mostly reproduced (0.331 observed
against a null median of 0.297), so A1 measures run dependence, not misfit; see the paper, Section VII-A];
null prevalence 0.001; gender 0.067 [0.058, 0.076], placebo 0.001 at N = 600;
uniform-shift power >= 0.8 on 21/25 items; prevalence 0.72-1.00 across random anchor sets.

## Airtightness review (3 October, evening) -- changes the headline
The 75 runs of a model are re-runs of one entity, not 75 examinees. Under a
replicated-human null (8 invariant 2PL humans, each answering once, re-run like the
models), the published pipeline flags 0.75 on average (scripts/a13_replicated_human_null25.py).
Model-as-unit analyses:
    python3 scripts/a11_person_fit25.py        # person fit, each model as ONE examinee: no model misfits
    python3 scripts/a12_eight_examinees25.py   # item test with 8 models as examinees: 5/25 (0.20) vs 0.006 for 8 real humans
    python3 scripts/a12_variants25.py          # 22 specifications: 0.04-0.32, VR.36 in all
    python3 scripts/a12_power25.py             # power with 8 examinees (0.25 at a 2-logit shift)
Run-level robustness (stable, but run-level): audit_scoring25.py (needs the restricted item text and full
logs; its outputs are shipped), make_variants25.py, robust_runner25.py,
robust_model25.py (domain | refboot | guess), a2_a10_25.py.
Library fix: dif/anchored.py now multi-starts the (a, b) fit and never accepts a value
below the nested b-only fit (stalled once, on VR.11). Published 0.84 is unchanged by it.
