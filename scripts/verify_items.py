"""Provenance check for the ICAR item file.

Runs before any administration. Verifies that the transcribed items agree with
(a) the published ICAR scoring key as shipped in psychTools, and (b) an
independent re-solve of the deterministic series items. A transcription error
here would mis-score the study silently, so this is a gate, not a report.
"""
from __future__ import annotations
import os, sys, json, string
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from harness.items import load_icar_text

ROOT = os.path.join(os.path.dirname(__file__), "..")
A = string.ascii_uppercase


def solve_series(stem):
    """Independently solve an alphanumeric series item from its stem."""
    body = stem.split("?", 1)[1]
    toks = [t.strip() for t in body.replace("...", "").split(",") if t.strip()]
    if not all(len(t) == 1 and t in A for t in toks):
        return None                       # numeric series: handled separately
    idx = [A.index(t) for t in toks]
    diffs = [(idx[i + 1] - idx[i]) % 26 for i in range(len(idx) - 1)]
    for period in (1, 2, 3):              # constant, alternating, or 3-cycle step
        if len(diffs) > period and all(
                diffs[i] == diffs[i % period] for i in range(len(diffs))):
            return A[(idx[-1] + diffs[len(diffs) % period]) % 26]
    d2 = [(diffs[i + 1] - diffs[i]) % 26 for i in range(len(diffs) - 1)]
    if len(set(d2)) == 1:                 # accelerating
        return A[(idx[-1] + (diffs[-1] + d2[0])) % 26]
    return None


def main():
    meta = pd.read_csv(os.path.join(ROOT, "data/derived/icar_items.csv"))
    kpath = os.environ.get("ICAR16_KEYS")          # published psychTools key: item,key CSV (data/README.md)
    if not kpath or not os.path.exists(kpath):
        sys.exit("verify_items needs ICAR16_KEYS (see data/README.md)")
    _k = pd.read_csv(kpath); pub = dict(zip(_k["item"], _k["key"]))
    pval = dict(zip(meta["item"], meta["p_value"]))
    # Item text is not distributed (PsychArchives Scientific Use Licence): set ICAR_ITEMS_8 to your
    # transcription of the 8 ICAR-16 text items (schema in data/README.md).
    path = os.environ.get("ICAR_ITEMS_8")
    if not path or not os.path.exists(path):
        sys.exit("verify_items needs ICAR_ITEMS_8 (see data/README.md)")
    items, awaiting, errs = load_icar_text(
        path,
        os.path.join(ROOT, "data/derived/icar_items.csv"), strict=False)

    checks, rows = {}, []
    checks["file validates"] = not errs
    checks["all 8 text items present"] = len(items) == 8
    key_ok = all(it.key == pub[it.item_id] for it in items)
    checks["keys match published psychTools key"] = key_ok

    solved = attempted = 0
    for it in items:
        ans = it.options[it.key - 1]
        indep = solve_series(it.stem) if "series" in it.domain else None
        if indep is not None:
            attempted += 1
            solved += (indep == ans)
        rows.append((it.item_id, it.key, pub[it.item_id], ans,
                     indep or "-", pval[it.item_id], len(it.options), it.n_fixed_tail))
    checks[f"series items independently re-solved ({solved}/{attempted})"] = (
        attempted > 0 and solved == attempted)
    checks["fixed tail on every item"] = all(it.n_fixed_tail == 2 for it in items)
    checks["key never points at fixed tail"] = all(
        it.key <= len(it.options) - it.n_fixed_tail for it in items)
    checks["8 options per item"] = all(len(it.options) == 8 for it in items)

    print(f"{'item':11s} {'key':>3s} {'pub':>3s} {'answer':<26s} {'re-solved':<10s} "
          f"{'human p':>7s} {'opts':>4s} {'tail':>4s}")
    for r in rows:
        print(f"{r[0]:11s} {r[1]:>3d} {r[2]:>3d} {r[3]:<26s} {r[4]:<10s} "
              f"{r[5]:>7.3f} {r[6]:>4d} {r[7]:>4d}")
    print()
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    bad = sum(1 for v in checks.values() if not v)
    print("\nRESULT:", "PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
