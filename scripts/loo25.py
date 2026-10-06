"""Leave-one-model-out DIF rate on the 25-item analysis."""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from multiprocessing import Pool
from dif.anchored import anchored_dif
from scripts.sapa_common import load_machine, load_reference_params, ITEMS, RES
path = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "main"
R_m, meta, _, _ = load_machine(path); a, b = load_reference_params()
models = sorted(meta.model.unique())
def run(m):
    keep = (meta.model != m).values
    r = anchored_dif(np.zeros((2, 25)), R_m[keep], item_names=ITEMS, ref_params=(a, b))
    return m, r.dif_rate, r.anchors, r.purify_status
if __name__ == "__main__":
    t = time.time()
    with Pool(4) as p: res = p.map(run, models)
    for r in res: print(f"{r[1]:.2f}  {r[3]:9s} anchors={r[2]}  drop {r[0]}")
    rates = [r[1] for r in res]
    print(f"LOO range {min(rates):.2f}-{max(rates):.2f}  ({time.time()-t:.0f}s)")
    json.dump([dict(dropped=r[0], rate=r[1], anchors=r[2], status=r[3]) for r in res],
              open(os.path.join(RES, f"loo25_{tag}.json"), "w"), indent=2)
