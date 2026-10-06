"""Study D (mechanism) and matched-N subgroup baselines.

Studies A-C revealed that the anchored estimator's Type-I rate inflates once the
focal sample becomes large relative to the reference sample -- the reverse of the
usual small-N concern.  The hypothesis is that this is anchor-parameter
uncertainty: reference item parameters are treated as known, so as focal N grows
the test acquires power to detect the reference sample's own estimation error.

Study D tests that directly by re-running the focal-N sweep with the TRUE item
parameters supplied instead of estimated ones.  If the inflation disappears, the
mechanism is confirmed and the remedy is a design constraint rather than a bug.

The second half computes the human-subgroup DIF baseline at MATCHED focal N, so
that placebo, subgroup, and machine rates are all measured in the same regime.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from multiprocessing import Pool
from dif.anchored import anchored_dif
from dif.mh import mantel_haenszel
from sim_core import simulate

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
DER = os.path.join(os.path.dirname(__file__), "..", "data", "derived")
REPS = int(os.environ.get("DREPS", 40))


# ----------------------------------------------------------- Study D
def _cell_D(kw):
    oracle = kw.pop("oracle")
    Rr, Rf, truth, (a, b, af, bf) = simulate(**kw)
    res = anchored_dif(Rr, Rf, purify=True,
                       ref_params=(a, b) if oracle else None)
    flag = res.table["flag_total"].values
    is_dif = truth != "none"
    return dict(n_foc=kw["n_foc"], oracle=oracle, seed=kw["seed"],
                power=float((flag & is_dif).sum() / max(is_dif.sum(), 1)),
                type1=float((flag & ~is_dif).sum() / max((~is_dif).sum(), 1)))


def study_d():
    base = dict(I=20, n_ref=2000, prop_dif=0.20, b_shift=0.8, a_ratio=0.45)
    cells = []
    for n in (200, 400, 800, 1600, 3200):
        for oracle in (True, False):
            for r in range(REPS):
                kw = dict(base); kw.update(n_foc=n, oracle=oracle, seed=770000 + n * 31 + r)
                cells.append(kw)
    t0 = time.time()
    with Pool(2) as p:
        rows = p.map(_cell_D, cells, chunksize=4)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "sim_D_anchor_uncertainty.csv"), index=False)
    print(f"[D] {len(d)} replicates in {time.time()-t0:.0f}s")
    print(d.groupby(["oracle", "n_foc"])[["power", "type1"]].mean().round(4).to_string())
    return d


# ------------------------------------------- matched-N subgroup baseline
def _cell_sub(args):
    R, ga, gb, n_foc, seed, names = args
    rng = np.random.default_rng(seed)
    ia = rng.permutation(np.where(ga)[0])
    ib = rng.permutation(np.where(gb)[0])[:n_foc]
    try:
        res = anchored_dif(R[ia], R[ib], item_names=names, purify=True)
        mh = mantel_haenszel(R[ia], R[ib], item_names=names)
        s = res.summary()
        s.update(seed=seed, n_foc=n_foc, n_ref=len(ia),
                 ets_BC_rate=float(((mh["ets"] == "B") | (mh["ets"] == "C")).mean()))
        return s
    except Exception as e:
        return dict(seed=seed, n_foc=n_foc, error=str(e))


def matched_subgroup(n_foc=300, reps=REPS):
    jobs, labels = [], []
    for tag, respath, demopath, col in (
            ("bfi", "bfi_responses.npy", "bfi_demo.csv", "gender"),
            ("spi", "spi_responses.npy", "spi_demo.csv", "sex")):
        R = np.load(os.path.join(DER, respath))
        demo = pd.read_csv(os.path.join(DER, demopath))
        g = demo[col].values
        vals = [v for v in np.unique(g[~pd.isna(g)])][:2]
        ga, gb = (g == vals[0]), (g == vals[1])
        names = [f"{tag}{i}" for i in range(R.shape[1])]
        for r in range(reps):
            jobs.append((R, ga, gb, n_foc, 660000 + hash(tag) % 1000 * 100 + r, names))
            labels.append(f"{tag}:{col}")
    t0 = time.time()
    with Pool(2) as p:
        rows = p.map(_cell_sub, jobs, chunksize=1)
    d = pd.DataFrame([dict(r, contrast=l) for r, l in zip(rows, labels) if "error" not in r])
    d.to_csv(os.path.join(OUT, "subgroup_matchedN.csv"), index=False)
    print(f"[matched subgroup] {len(d)} runs in {time.time()-t0:.0f}s  (focal N={n_foc})")
    print(d.groupby("contrast")[["dif_rate_total", "dif_rate_uniform",
                                 "dif_rate_nonuniform", "ets_BC_rate"]]
          .agg(["mean", "std"]).round(4).to_string())
    return d


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "D"): study_d()
    if which in ("all", "sub"): matched_subgroup()
