"""A13: the right null for analyses that treat a model's runs as respondents.

Eight synthetic humans, each at one model's ability (theta from its modal answers,
as in A12), answer the 25 items ONCE from the SAPA 2PL; each answer vector is then
re-run as that model was (75 runs, the model's real missing cells), flipping each
answer with that model's own observed run-to-run inconsistency. Everything is
invariant by construction. The published pipeline (anchored DIF) and the A1
statistic are run on each of 200 such data sets.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
from dif.anchored import anchored_dif
from dif.irt import prob_2pl
import scripts.a1_model_fit25 as A1
from scripts.sapa_common import load_machine, load_reference_params, ITEMS, RES
A, B = load_reference_params()
R_m, meta, _, _ = load_machine("data/responses_free25.jsonl")
MODELS = sorted(meta.model.unique())
TH = json.load(open(os.path.join(RES, "a12_eight_examinees25.json")))["model_thetas"]
MASKS, FLIP = [], []
for m in MODELS:
    Rm = R_m[(meta.model == m).values]; MASKS.append(~np.isnan(Rm))
    P = np.nanmean(Rm, 0); pq = np.nanmean(P * (1 - P)); FLIP.append((1 - np.sqrt(1 - 4 * pq)) / 2)

def one(rep):
    rng = np.random.default_rng(130000 + rep); blocks = []
    for m, M, f in zip(MODELS, MASKS, FLIP):
        base = (rng.random(25) < prob_2pl([TH[m]], A, B)[0]).astype(float)
        Y = np.tile(base, (M.shape[0], 1)); Y = np.where(rng.random(M.shape) < f, 1 - Y, Y); Y[~M] = np.nan
        blocks.append(Y)
    R = np.vstack(blocks)
    r = anchored_dif(np.zeros((2, 25)), R, item_names=ITEMS, ref_params=(A, B))
    g = [A1.stats_of(b_, True) for b_ in blocks]
    return dict(rep=rep, rate=r.dif_rate, sigma=r.focal_sigma, rmsd_min=min(x["rmsd"] for x in g),
                rmsd_med=float(np.median([x["rmsd"] for x in g])), rmsd_max=max(x["rmsd"] for x in g),
                G2_med=float(np.median([x["G2"] for x in g])))

if __name__ == "__main__":
    print("flip rates", dict(zip([m.split("/")[1] for m in MODELS], np.round(FLIP, 3))), flush=True)
    with Pool(4) as p: out = pd.DataFrame(p.map(one, range(int(os.environ.get("REPS", 200))), chunksize=2))
    out.to_csv(os.path.join(RES, "a13_replicated_human_null25.csv"), index=False)
    s = dict(reps=len(out), rate_mean=out.rate.mean(), rate_q05=out.rate.quantile(.05), rate_q95=out.rate.quantile(.95),
             p_rate_ge_084=float((out.rate >= 0.84).mean()), sigma0_share=float((out.sigma < 1e-3).mean()),
             rmsd_med_median=out.rmsd_med.median(), rmsd_med_q95=out.rmsd_med.quantile(.95),
             observed_rmsd_median=0.3311)
    json.dump(s, open(os.path.join(RES, "a13_replicated_human_null25.json"), "w"), indent=2, default=float)
    print(json.dumps(s, indent=2, default=float))
