"""Leave-two-models-out DIF rate (28 runs)."""
import os, sys, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from multiprocessing import Pool
from dif.anchored import anchored_dif
from scripts.sapa_common import load_machine, load_reference_params, ITEMS, RES
path = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "main"
R_m, meta, _, _ = load_machine(path); a, b = load_reference_params()
models = sorted(meta.model.unique())
def run(pair):
    keep = (~meta.model.isin(pair)).values
    r = anchored_dif(np.zeros((2, 25)), R_m[keep], item_names=ITEMS, ref_params=(a, b))
    return dict(dropped=list(pair), rate=r.dif_rate, status=r.purify_status)
if __name__ == "__main__":
    with Pool(4) as p: res = p.map(run, list(itertools.combinations(models, 2)))
    rates = np.array([r["rate"] for r in res])
    print(f"L2O: n={len(rates)} min {rates.min():.2f} median {np.median(rates):.2f} max {rates.max():.2f}")
    json.dump(res, open(os.path.join(RES, f"l2o25_{tag}.json"), "w"), indent=2)
