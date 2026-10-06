"""Run the primary anchored DIF and the A1 model-level test on any response file.
python3 scripts/robust_runner25.py <jsonl> <tag> [items=all|verbal|series]"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
from dif.anchored import anchored_dif
import scripts.a1_model_fit25 as A1
from scripts.sapa_common import load_machine, ITEMS, RES

def a1_one(args):
    m, R, seed, Bn = args
    rng = np.random.default_rng(seed)
    s = A1.stats_of(R, True); bs = A1.boot(R, s["mu"], s["sigma"], Bn, rng, True)
    return dict(model=m, G2=s["G2"], df=s["df"], q95=float(np.quantile(bs, .95)),
                p=float((1 + (bs >= s["G2"]).sum()) / (Bn + 1)), rmsd=s["rmsd"], mu=s["mu"], sigma=s["sigma"])

def run(path, tag, Bn=500, do_a1=True):
    R, meta, long, excl = load_machine(path, min_parse=0.0)
    r = anchored_dif(np.zeros((2, 25)), R, item_names=ITEMS, ref_params=(A1.A, A1.B))
    out = dict(tag=tag, n_resp=int(R.shape[0]), acc=float(np.nanmean(R)), parse=float(long.parsable.mean()),
               dif_rate=r.dif_rate, n_flag=int(r.table.flag_total.sum()), mu=r.focal_mu, sigma=r.focal_sigma,
               invariant=[ITEMS[i] for i in np.where(~r.table.flag_total.values)[0]], status=r.purify_status)
    if do_a1:
        models = sorted(meta.model.unique())
        jobs = [(m, R[(meta.model == m).values], 200 + k, Bn) for k, m in enumerate(models)]
        with Pool(4) as p: res = p.map(a1_one, jobs)
        a = pd.DataFrame(res)
        out.update(a1_reject_001=int((a.p <= 0.002).sum()), a1_n=len(a), a1_rmsd_min=float(a.rmsd.min()),
                   a1_rmsd_max=float(a.rmsd.max()), a1_G2_over_q95_min=float((a.G2 / a.q95).min()))
    return out

if __name__ == "__main__":
    path, tag = sys.argv[1], sys.argv[2]
    o = run(path, tag); print(json.dumps(o, default=float))
    with open(os.path.join(RES, "robustness25.jsonl"), "a") as f: f.write(json.dumps(o, default=float) + "\n")
