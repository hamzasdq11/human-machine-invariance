"""C12: real-human groups of eight, matched to the models' abilities (7 Oct 2026).

c7 draws, for each item, eight SAPA respondents who answered it and >= 5 other items. Those groups
average theta near +0.2, while the models sit at -0.93 to -2.08, and a person's ability rests on
about seven items (median) against a model's 24. This script repeats c7's calibration with each group matched
to the models: for item i, model m contributes one SAPA respondent whose ability, estimated from his
or her other items exactly as in the test, lies within CALIPER logits of model m's theta (the
model-as-examinee theta of Table models). Pools with fewer than MIN_POOL candidates are widened to
the MIN_POOL nearest. Everything else is as in c7: exact two-sided Poisson-binomial test (comparator
for the modal and single-run reductions), stop-loss bound (comparator for the run mean), BH at 0.05
over the 25 items, p = (1 + k) / (NREP + 1).

The match fixes estimated ability, not true ability: a person whose seven or so items put him or her at
-1.8 is, on average, abler than that (regression to the mean), and that is part of what the
calibration has to absorb, as it is for the models' own abilities.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import common as C

rng = np.random.default_rng(20261007)
NREP = int(os.environ.get("NREP", 1000))
CALIPER = float(os.environ.get("CALIPER", 0.25))
MIN_POOL = int(os.environ.get("MIN_POOL", 30))
OBS = dict(modal=0.20, single=0.12, run_mean=0.08)

SAPA = os.path.join(C.ROOT, "data", "sapaICARData18aug2010thru20may2013.csv")
d = pd.read_csv(SAPA, index_col=0)
X = d[C.ITEMS].to_numpy(float)
X = X[~np.isnan(X).all(1)]
O = ~np.isnan(X)
nit = O.sum(1)

pf = pd.read_csv(os.path.join(C.ROOT, "results", "a11_person_fit25.csv"))
TH_MODEL = np.sort(pf["theta"].to_numpy(float))

G = np.linspace(-4, 4, 1601)
ZG = C.A * (G[:, None] - C.B)
LPG, LQG = -np.logaddexp(0, -ZG), -np.logaddexp(0, ZG)


def theta_rows(M, chunk=4000):
    """ML theta for each row of M (0/1/NaN), bounded [-4, 4]: grid search + Newton (as in c7)."""
    out = np.empty(len(M))
    for s in range(0, len(M), chunk):
        Mc = M[s:s + chunk]
        Om = ~np.isnan(Mc); M0 = np.where(Om, Mc, 0.0)
        LL = M0 @ LPG.T + (Om - M0) @ LQG.T
        th = G[LL.argmax(1)]
        for _ in range(6):
            P = 1 / (1 + np.exp(-C.A * (th[:, None] - C.B)))
            sc = (Om * C.A * (M0 - P)).sum(1); inf = (Om * C.A ** 2 * P * (1 - P)).sum(1)
            th = np.clip(th + np.clip(np.where(inf > 0, sc / np.maximum(inf, 1e-12), 0), -.5, .5), -4, 4)
        out[s:s + chunk] = th
    return out


def run(pool_mask, label):
    cand, th_other = [], []
    for i in range(25):
        rows = np.where(pool_mask & O[:, i])[0]
        M = X[rows].copy(); M[:, i] = np.nan
        cand.append(rows); th_other.append(theta_rows(M))
    # matched candidate sets: per item, per model
    sets, widened = [], 0
    for i in range(25):
        si = []
        for t in TH_MODEL:
            dist = np.abs(th_other[i] - t)
            idx = np.where(dist <= CALIPER)[0]
            if len(idx) < MIN_POOL:
                idx = np.argsort(dist)[:MIN_POOL]; widened += 1
            si.append(idx)
        sets.append(si)
    pool_sizes = [len(s) for si in sets for s in si]
    F_pb, F_sl = np.zeros((NREP, 25), bool), np.zeros((NREP, 25), bool)
    th_used = []
    for k in range(NREP):
        p_pb, p_sl = np.ones(25), np.ones(25)
        for i in range(25):
            chosen = []
            for m in range(8):
                for _ in range(20):                      # one person per model, no repeats within the item
                    j = int(rng.choice(sets[i][m]))
                    if j not in chosen:
                        break
                chosen.append(j)
            chosen = np.array(chosen)
            th = th_other[i][chosen]; th_used.append(th.mean())
            y = X[cand[i][chosen], i]
            pr = 1 / (1 + np.exp(-C.A[i] * (th - C.B[i])))
            p_pb[i] = C.pb_p_two_sided(pr, int(y.sum()))
            p_sl[i] = C.stoploss_p_two_sided(pr, float(y.sum()))
        F_pb[k], F_sl[k] = C.bh(p_pb), C.bh(p_sl)
    rp, rs = F_pb.mean(1), F_sl.mean(1)

    def pval(r, obs):
        return float((1 + (r >= obs - 1e-9).sum()) / (NREP + 1))

    res = dict(pool=label, caliper=CALIPER, min_pool=MIN_POOL, widened_sets=widened,
               pool_size_median=float(np.median(pool_sizes)), pool_size_min=int(np.min(pool_sizes)),
               theta_models=[round(float(t), 3) for t in TH_MODEL],
               theta_matched_group_mean=float(np.mean(th_used)),
               exact_pb=dict(mean=float(rp.mean()), q95=float(np.quantile(rp, .95)), max=float(rp.max()),
                             share_ge_020=float((rp >= 0.20 - 1e-9).mean()),
                             p_modal_020=pval(rp, OBS["modal"]), p_single_012=pval(rp, OBS["single"]),
                             item_rate=dict(zip(C.ITEMS, np.round(F_pb.mean(0), 4).tolist()))),
               stoploss=dict(mean=float(rs.mean()), q95=float(np.quantile(rs, .95)), max=float(rs.max()),
                             p_runmean_008=pval(rs, OBS["run_mean"])))
    print(label, json.dumps({k: v for k, v in res.items() if k not in ("exact_pb",)}, default=float), flush=True)
    print("  exact_pb", {k: v for k, v in res["exact_pb"].items() if k != "item_rate"}, flush=True)
    return res


# unmatched reference, for the record: mean theta of c7-style groups
ref_th = []
for i in range(25):
    rows = np.where((nit >= 6) & O[:, i])[0]
    M = X[rows].copy(); M[:, i] = np.nan
    ref_th.append(float(theta_rows(M).mean()))
out = dict(unmatched_pool_mean_theta_by_item=dict(zip(C.ITEMS, np.round(ref_th, 3).tolist())),
           unmatched_pool_mean_theta=float(np.mean(ref_th)),
           matched=dict(ge5_other=run(nit >= 6, "ge5_other"), ge8_other=run(nit >= 9, "ge8_other")),
           nrep=NREP)
C.save_json(out, "c12_matched_humans.json")
