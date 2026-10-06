"""Human-side control analyses: the placebo floor and the subgroup calibration.

Two of the four controls the main claim depends on, and neither needs a model:

  1. Random-split placebo.  Split a single human sample arbitrarily and run the
     identical pipeline.  A pipeline that manufactures DIF will show it here.
     This bounds the false-positive floor of the whole procedure on real data.

  2. Human-subgroup DIF.  Real demographic contrasts on real instruments.  This
     establishes what ordinary cross-population non-invariance looks like, which
     is the yardstick the human-vs-machine result must be read against.
"""
from __future__ import annotations
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from multiprocessing import Pool
from dif.anchored import anchored_dif
from dif.mh import mantel_haenszel
from dif.irt import fit_2pl_mml

DER = os.path.join(os.path.dirname(__file__), "..", "data", "derived")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
N_PLACEBO = int(os.environ.get("N_PLACEBO", 150))


def _placebo_one(args):
    R, n_foc, seed, names = args
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(R))
    foc, ref = idx[:n_foc], idx[n_foc:]
    try:
        res = anchored_dif(R[ref], R[foc], item_names=names, purify=True)
        s = res.summary(); s.update(seed=seed, n_foc=n_foc, n_ref=len(ref))
        return s
    except Exception as e:
        return dict(seed=seed, n_foc=n_foc, error=str(e))


def placebo(R, names, sizes=(100, 250, 500, 762), reps=N_PLACEBO, tag="icar"):
    jobs = [(R, n, 900000 + n * 1000 + r, names) for n in sizes for r in range(reps)]
    t0 = time.time()
    with Pool(2) as p:
        rows = p.map(_placebo_one, jobs, chunksize=2)
    df = pd.DataFrame([r for r in rows if "error" not in r])
    df.to_csv(os.path.join(OUT, f"placebo_{tag}.csv"), index=False)
    print(f"[placebo:{tag}] {len(df)} splits in {time.time()-t0:.0f}s")
    print(df.groupby("n_foc")[["dif_rate_total", "dif_rate_uniform",
                               "dif_rate_nonuniform"]].agg(["mean", "std"]).round(4).to_string())
    return df


def subgroup(R, names, group, label, tag, min_n=200):
    g = np.asarray(group)
    ok = ~pd.isna(g)
    vals, counts = np.unique(g[ok], return_counts=True)
    vals = [v for v, c in zip(vals, counts) if c >= min_n]
    if len(vals) < 2:
        print(f"[subgroup:{tag}:{label}] insufficient group sizes"); return None
    a, b = vals[0], vals[1]
    Ra, Rb = R[ok][g[ok] == a], R[ok][g[ok] == b]
    res = anchored_dif(Ra, Rb, item_names=names, purify=True)
    mh = mantel_haenszel(Ra, Rb, item_names=names)
    s = res.summary()
    s.update(instrument=tag, contrast=f"{label}:{a}v{b}", n_ref=len(Ra), n_foc=len(Rb),
             ets_B=int((mh["ets"] == "B").sum()), ets_C=int((mh["ets"] == "C").sum()),
             ets_BC_rate=float(((mh["ets"] == "B") | (mh["ets"] == "C")).mean()),
             mh_sig_rate=float(mh["sig"].mean()))
    res.table.assign(**{"ets": mh["ets"].values, "delta_mh": mh["delta_mh"].values}) \
        .to_csv(os.path.join(OUT, f"subgroup_{tag}_{label}.csv"), index=False)
    print(f"[subgroup:{tag}:{label}] N={len(Ra)}/{len(Rb)} "
          f"LR-DIF={s['dif_rate_total']:.3f} MH B+C={s['ets_BC_rate']:.3f}")
    return s


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    meta = pd.read_csv(os.path.join(DER, "icar_items.csv"))
    R = np.load(os.path.join(DER, "icar_responses.npy"))
    names = meta["item"].tolist()

    if which in ("all", "calib"):
        fit = fit_2pl_mml(R, names=names)
        tab = fit.as_frame().merge(meta, on="item")
        tab.to_csv(os.path.join(OUT, "icar_calibration.csv"), index=False)
        print(f"[calibration] converged={fit.converged} iters={fit.n_iter} loglik={fit.loglik:.1f}")
        print(tab[["item", "domain", "p_value", "a", "b"]].round(3).to_string(index=False))

    if which in ("all", "placebo"):
        placebo(R, names, tag="icar")

    if which in ("all", "subgroup"):
        rows = []
        bfi = np.load(os.path.join(DER, "bfi_responses.npy"))
        bdemo = pd.read_csv(os.path.join(DER, "bfi_demo.csv"))
        bnames = [f"bfi{i}" for i in range(bfi.shape[1])]
        rows.append(subgroup(bfi, bnames, bdemo["gender"], "gender", "bfi"))
        rows.append(subgroup(bfi, bnames, (bdemo["age"] > bdemo["age"].median()).astype(float),
                             "agesplit", "bfi"))
        spi = np.load(os.path.join(DER, "spi_responses.npy"))
        sdemo = pd.read_csv(os.path.join(DER, "spi_demo.csv"))
        snames = [f"spi{i}" for i in range(spi.shape[1])]
        rows.append(subgroup(spi, snames, sdemo["sex"], "sex", "spi"))
        rows.append(subgroup(spi, snames, (sdemo["education"] > sdemo["education"].median()).astype(float),
                             "edusplit", "spi"))
        pd.DataFrame([r for r in rows if r]).to_csv(
            os.path.join(OUT, "subgroup_summary.csv"), index=False)
