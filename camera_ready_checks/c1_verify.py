"""C1: independent re-derivation of the draft's model-as-examinee numbers.

Checks, from the raw response files and the SAPA item parameters only:
  respondent accounting (Section V), modal answers and theta (Table models),
  the item test (Table unit), the temperature-0 single answers, lz* person fit,
  the simulated 2PL null of the item test, and per-item power.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import common as C

rng = np.random.default_rng(20261005)
main = C.load_long(os.path.join(C.ROOT, "data", "responses_free25.jsonl"))
tz = C.load_long(os.path.join(C.ROOT, "data", "responses_free25_temp0.jsonl"))
models = sorted(main.model.unique())
out = {}

# ---- accounting
R, meta = C.run_matrix(main)
acc = dict(rows=len(main), models=len(models), runs=len(meta),
           runs_per_model=meta.groupby("model").size().unique().tolist(),
           frames=sorted(meta.frame.unique()), seeds=int(meta.seed.nunique()),
           parsed=int(main.parsable.sum()), parse_rate=float(main.parsable.mean()),
           accuracy=float(np.nanmean(main.y)),
           per_item_parsed_n_median=float(np.median((~np.isnan(R)).sum(0))),
           per_item_parsed_n_range=[int((~np.isnan(R)).sum(0).min()), int((~np.isnan(R)).sum(0).max())],
           parse_by_frame=main.groupby("frame").parsable.mean().round(3).to_dict(),
           temp0_rows=len(tz), temp0_runs=int(tz.groupby(["model", "frame", "seed"]).ngroups),
           temp0_parse=float(tz.parsable.mean()), temp0_accuracy=float(np.nanmean(tz.y)))
out["accounting"] = acc

# ---- modal answers, theta, lz*
P, N, _ = C.model_props(main, models)
X = C.modal(P)
rows = []
for m, name in enumerate(models):
    obs = np.where(~np.isnan(X[m]))[0]
    lz, th6 = C.lz_star(X[m, obs], obs)                    # theta on [-6, 6] as in a11
    th4 = C.theta_ml(X[m, obs], obs)                        # [-4, 4] as in a12
    Ps = C.p2pl(th6)[obs]
    sims = []
    for _ in range(4000):
        xs = (rng.random(len(obs)) < Ps).astype(float)
        v, _ = C.lz_star(xs, obs)
        if not np.isnan(v): sims.append(v)
    rows.append(dict(model=C.short(name), items=len(obs), ties=int(np.sum(np.abs(P[m] - .5) < 1e-12)),
                     theta=th4, lz_star=lz, p_lz_sim=float(np.mean(np.array(sims) <= lz)),
                     accuracy=float(np.nanmean(main[main.model == name].y)),
                     parse=float(main[main.model == name].parsable.mean())))
pf = pd.DataFrame(rows); pf.to_csv(os.path.join(C.OUT, "c1_models.csv"), index=False)
out["person_fit"] = pf.round(4).to_dict("records")

# ---- item test (modal answers) and temperature-0 single answers (v1, seed 0)
t = C.fast_item_test(X); t.to_csv(os.path.join(C.OUT, "c1_item_test_modal.csv"), index=False)
out["item_test_modal"] = dict(flagged=list(t.item[t.flag]), rate=float(t.flag.mean()))
one = tz[(tz.frame == "v1") & (tz.seed == 0)]
X0 = one.pivot_table(index="model", columns="item", values="y", aggfunc="first").reindex(index=models, columns=C.ITEMS).to_numpy(float)
t0 = C.fast_item_test(X0)
out["item_test_temp0_v1s0"] = dict(flagged=list(t0.item[t0.flag]), rate=float(t0.flag.mean()))
Pz, _, _ = C.model_props(tz, models)
tz_modal = C.fast_item_test(C.modal(Pz)); tz_modal.to_csv(os.path.join(C.OUT, "c1_item_test_temp0_modal.csv"), index=False)
out["item_test_temp0_modal"] = dict(flagged=list(tz_modal.item[tz_modal.flag]), rate=float(tz_modal.flag.mean()))

# ---- simulated 2PL null at the models' thetas, models' missing cells
TH = np.array([r["theta"] for r in rows])
PH = C.p2pl(TH)
rates, fl = [], np.zeros(25)
for _ in range(4000):
    Xs = (rng.random((8, 25)) < PH).astype(float); Xs[np.isnan(X)] = np.nan
    f = C.fast_item_test(Xs).flag.to_numpy(); rates.append(f.mean()); fl += f
rates = np.array(rates)
out["sim_null"] = dict(reps=len(rates), mean=rates.mean(), q95=float(np.quantile(rates, .95)), max=rates.max(),
                       share_any_flag=float((rates > 0).mean()), share_ge_observed=float((rates >= 0.2).mean()))

# ---- per-item power: ONE item shifted at a time, by d logits, for all 8 examinees
pw = []
for d in (1.0, 2.0, 3.0, 4.0):
    for direction in (-1, +1):                       # -1: easier for machines, +1: harder
        for i in range(25):
            bb = C.B.copy(); bb[i] += direction * d
            PH_alt = C.p2pl(TH, C.A, bb)
            hits = 0
            for _ in range(200):
                Xs = (rng.random((8, 25)) < PH_alt).astype(float); Xs[np.isnan(X)] = np.nan
                hits += C.fast_item_test(Xs).flag.to_numpy()[i]
            pw.append(dict(shift=d, direction="easier" if direction < 0 else "harder", item=C.ITEMS[i], power=hits / 200))
pw = pd.DataFrame(pw); pw.to_csv(os.path.join(C.OUT, "c1_power_one_item.csv"), index=False)
out["power_one_item"] = {f"{d}_{dr}": dict(median=float(g.power.median()), items_ge_08=int((g.power >= .8).sum()))
                         for (d, dr), g in pw.groupby(["shift", "direction"])}
C.save_json(out, "c1_verify.json")
print(json.dumps({k: v for k, v in out.items() if k != "person_fit"}, indent=1, default=str))
print(pf.round(3).to_string(index=False))
