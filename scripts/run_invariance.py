"""The primary analysis: human reference vs machine focal group.

Consumes the response matrix produced by the Colab harness and runs exactly the
analysis specified in PREREGISTRATION.md, with no further choices.  Every control
required by the pre-registration is computed here or read from a results file
produced earlier, so the outputs of this script are the paper's Results section.

    python3 scripts/run_invariance.py --machine machine_responses.parquet
"""
from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from dif.anchored import anchored_dif
from dif.mh import mantel_haenszel

DER = os.path.join(os.path.dirname(__file__), "..", "data", "derived")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def load_human(text_only=True, human=None):
    """ICAR-16 human responses. data/derived/icar_responses.npy is built by scripts/prep_data.py from
    the psychTools R package (GPL) and is not distributed here; pass another matrix with --human."""
    meta = pd.read_csv(os.path.join(DER, "icar_items.csv"))
    R = np.load(human or os.path.join(DER, "icar_responses.npy"))
    if text_only:
        keep = meta["text_administrable"].values.astype(bool)
        R, meta = R[:, keep], meta[keep].reset_index(drop=True)
    return R, meta


def align(machine_wide, item_names):
    """Reorder machine columns to the human item order; missing items -> NaN."""
    M = machine_wide.reindex(columns=list(item_names))
    missing = [c for c in item_names if c not in machine_wide.columns]
    if missing:
        print(f"  [warn] machine matrix missing {len(missing)} items: {missing[:5]}")
    return M.to_numpy(dtype=float)


def variance_components(meta_df, wide):
    """Facet variance decomposition justifying the respondent construction.

    Reports how much of the variance in respondent-level score is attributable to
    model identity, prompt variant, and seed.  Required by the pre-registration:
    the (model, prompt, seed) respondent is a modelling choice and its
    consequences are quantified rather than assumed.
    """
    import statsmodels.formula.api as smf
    d = meta_df.copy()
    d["score"] = wide.mean(axis=1).values
    d = d.dropna(subset=["score"])
    m = smf.ols("score ~ C(model) + C(prompt_variant) + C(seed)", data=d).fit()
    tot = d["score"].var(ddof=1)
    comp = {}
    for fac in ("model", "prompt_variant", "seed"):
        red = smf.ols(f"score ~ {' + '.join('C(%s)' % f for f in ('model','prompt_variant','seed') if f != fac)}",
                      data=d).fit()
        comp[fac] = max(m.rsquared - red.rsquared, 0.0)
    comp["residual"] = max(1 - m.rsquared, 0.0)
    comp["total_var"] = float(tot)
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, help="parquet from the harness")
    ap.add_argument("--meta", default=None, help="machine_meta.csv")
    ap.add_argument("--min-parse", type=float, default=0.90)
    ap.add_argument("--all-items", action="store_true",
                    help="use all 16 ICAR items (vision-capable respondents only)")
    ap.add_argument("--human", default=None, help="human response matrix (.npy, columns as icar_items.csv)")
    ap.add_argument("--out", default=OUT, help="output directory (default: results/)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    R_h, meta = load_human(text_only=not a.all_items, human=a.human)
    names = meta["item"].tolist()
    wide = pd.read_parquet(a.machine)

    mdf = pd.read_csv(a.meta, index_col=0) if a.meta else None
    if mdf is not None:
        keep_models = mdf.groupby("model")["parsable_rate"].mean()
        excluded = list(keep_models[keep_models < a.min_parse].index)
        if excluded:
            print(f"[exclusion] parse rate < {a.min_parse}: {excluded}")
            wide = wide.loc[~mdf.loc[wide.index, "model"].isin(excluded)]
            mdf = mdf.loc[wide.index]

    R_m = align(wide, names)
    print(f"[data] human {R_h.shape}  machine {R_m.shape}  items {len(names)}")

    res = anchored_dif(R_h, R_m, item_names=names, purify=True, verbose=True)
    mh = mantel_haenszel(R_h, R_m, item_names=names)
    tab = res.table.assign(ets=mh["ets"].values, delta_mh=mh["delta_mh"].values,
                           domain=meta["domain"].values)
    tab["uniformity_index"] = (tab["lr_uniform"] / tab["lr_total"].clip(lower=1e-9)).clip(0, 1)
    tab.to_csv(os.path.join(a.out, "invariance_human_vs_machine.csv"), index=False)

    summ = res.summary()
    summ.update(ets_BC_rate=float(((mh["ets"] == "B") | (mh["ets"] == "C")).mean()),
                n_machine_respondents=int(R_m.shape[0]),
                n_human=int(R_h.shape[0]), item_set="text_only" if not a.all_items else "all")

    # invariance-purified contrast: does the comparison change on invariant items?
    inv = ~tab["flag_total"].values
    if inv.sum() >= 3:
        from dif.irt import eap_scores
        a_ref, b_ref = res.ref_params["a"].values, res.ref_params["b"].values
        th_h, _ = eap_scores(R_h[:, inv], a_ref[inv], b_ref[inv])
        th_m, _ = eap_scores(R_m[:, inv], a_ref[inv], b_ref[inv])
        raw_h, raw_m = np.nanmean(R_h, 1), np.nanmean(R_m, 1)
        summ.update(n_invariant_items=int(inv.sum()),
                    raw_gap=float(np.nanmean(raw_m) - np.nanmean(raw_h)),
                    purified_theta_gap=float(np.nanmean(th_m) - np.nanmean(th_h)))

    if mdf is not None:
        try:
            summ["variance_components"] = variance_components(mdf, wide)
        except Exception as e:
            print("  [warn] variance components unavailable:", e)

    with open(os.path.join(a.out, "invariance_summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=float)
    print(json.dumps(summ, indent=2, default=float))


if __name__ == "__main__":
    main()
