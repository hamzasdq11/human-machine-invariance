"""A3 (part 1): log the purification path of the published run.
Records, per iteration, the anchor set used, mu/sigma, the BH flag count, and
whether the 4-anchor floor replaced the unflagged set."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import dif.anchored as A
from scripts.sapa_common import load_machine, load_reference_params, ITEMS, RES
path = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "main"
R_m, meta, _, _ = load_machine(path); a, b = load_reference_params()
log = []; orig_bh = A.benjamini_hochberg; orig_est = A.estimate_group_dist
state = {}
def est(R, aa, bb, **kw):
    mu, s, ll = orig_est(R, aa, bb, **kw)
    state["anchors"] = [ITEMS[int(np.where(np.isclose(a, x) & np.isclose(b, y))[0][0])] for x, y in zip(aa, bb)]
    state["mu"], state["sigma"] = mu, s; state["calls"] = 0
    return mu, s, ll
def bh(p, alpha=0.05):
    rej = orig_bh(p, alpha)
    if state.get("calls", 0) == 0:   # first call per iteration is the total test
        log.append(dict(iteration=len(log) + 1, anchors_used=state["anchors"],
                        mu=state["mu"], sigma=state["sigma"], n_flagged=int(rej.sum()),
                        unflagged=[ITEMS[i] for i in np.where(~rej)[0]],
                        floor_binding=bool((~rej).sum() < 4)))
    state["calls"] = state.get("calls", 0) + 1
    return rej
A.benjamini_hochberg = bh; A.estimate_group_dist = est
r = A.anchored_dif(np.zeros((2, 25)), R_m, item_names=ITEMS, ref_params=(a, b))
for e in log: print(json.dumps(e, default=float))
print("final", r.anchors, r.purify_status, r.dif_rate)
json.dump(dict(path=log, final=r.anchors, status=r.purify_status), open(os.path.join(RES, f"anchor_path25_{tag}.json"), "w"), indent=2, default=float)
