# Changelog

## 1.1.1 (7 October 2026)

- Removes a letter series quoted in a comment in `camera_ready_checks/c10_exposure.py` and the published
  keys of the 16 ICAR sample-test items (`data/derived/icar_items.csv`, `results/icar_calibration.csv`,
  `scripts/prep_data.py`). `scripts/verify_items.py` now reads that key from a file you supply
  (`ICAR16_KEYS`).
- Archived on Zenodo as doi:10.5281/zenodo.23201160 (all versions: doi:10.5281/zenodo.23175503).

**Documentation**
- The August runbook and README are marked as lightly edited.
- The matched-group documentation says "about seven items", the SAPA median.

## 1.1.0 (7 October 2026)

**ICAR item material.** ICAR item text and keys come from the PsychArchives release
(doi:10.23668/psycharchives.22167) and are not redistributed. This version removes the item stems,
options and keys, and the model output text, option orders, presented keys and choices from the response
files, which keep scored outcomes only.

**Added**
- `data/items25_index.json`: identifiers, stem length and answer type, with no text.
- Scored lenient-parser variants of both arms.
- `scripts/make_variants25.py`.
- `camera_ready_checks/c12_matched_humans.py`: real groups of eight matched to the models' abilities.
- `c11b_bundle_regression_null.py`: the letter-series bundle under a null regressing to the human mean.
- `c13_macros.py`: macros for the paper's 7 October numbers.
- `CHANGELOG.md`.

**Fixed**
- The end-to-end test no longer writes synthetic output into `results/`, and pytest now finds it.
  `results/invariance_human_vs_machine.csv` and `results/invariance_summary.json` were outputs of that
  test, not results, and are removed.
- `RECOVERY.md`:
  - The "all 8 models reject" line from 3 October is marked as superseded.
  - The ETS B+C figure is marked as withdrawn.
- `results/SUMMARY.md` (August, 8-item phase) is removed as stale.
- The August runbook and README are moved to `docs/history/` with a note.
- The `psychTools` ICAR-16 response matrix (GPL) is no longer included. `scripts/prep_data.py` rebuilds it.
- The test now simulates its human reference from the published ICAR-16 calibration.
- `results/inv25_main_table.csv` and `results/inv25_main_summary.json` were re-run with the current
  library. The shipped files predated the multi-start fix recorded in `RECOVERY.md`. Flags, the 0.84 rate
  and the focal distribution are unchanged; the per-item focal (a, b) estimates differ, and so does the
  likelihood ratio for VR.11, where the old fit had stalled. The paper uses none of these. Other result
  files are as in 1.0.0.

No analysis result in the paper changed. Every analysis reads scored outcomes only, and re-running
`run_invariance25.py` and `c9` on the stripped files reproduces the full-log output exactly.

## 1.0.0 (6 October 2026)

Camera-ready snapshot.
