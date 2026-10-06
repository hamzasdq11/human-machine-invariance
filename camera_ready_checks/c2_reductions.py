"""C2: does the model-level item test depend on how a model's 75 runs are reduced?

Three reductions of a model's runs to one examinee:
  modal   : majority answer per item (the draft; ties dropped)
  single  : one randomly chosen run (frame, seed): literally "answering once"
  mean    : the model's proportion correct per item, tested with a stop-loss
            (convex-order) bound on the Poisson-binomial. Valid whatever the
            dependence among a model's runs, because a proportion in [0, 1] with
            mean P is never more dispersed than one Bernoulli(P) answer.

Each is calibrated under three nulls (8 synthetic humans at the models' thetas,
the models' run counts and missing cells; invariance holds in all three):
  A0  fixed answers  : one 2PL answer vector, copied to every run
  B   fresh answers  : every run an independent 2PL draw
  C   propensities   : Beta propensities, mean = 2PL, spread matched to each
                       model's observed within-cell variance (as in c4)
Each test assumes something different: modal is exact under A0 only, single
under B and C, mean under all three.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import common as C

rng = np.random.default_rng(5102026)
NDRAW = int(os.environ.get("NDRAW", 2000))
NNULL = int(os.environ.get("NNULL", 400))


def reduce_all(R, meta, models, rng):
    """R: run x item matrix -> (modal X, mean X, one random run X)."""
    mod, mean, single = [], [], []
    for m in models:
        Rm = R[(meta.model == m).values]
        p = np.nanmean(Rm, 0) if len(Rm) else np.full(25, np.nan)
        mean.append(p)
        mod.append(C.modal(p))
        single.append(Rm[rng.integers(len(Rm))])
    return np.array(mod), np.array(mean), np.array(single)


def tests(Xmod, Xmean, Xsingle):
    return dict(modal=C.fast_item_test(Xmod).flag.to_numpy(),
                mean=C.fast_item_test(Xmean, stoploss=True).flag.to_numpy(),
                single=C.fast_item_test(Xsingle).flag.to_numpy())


def run_arm(path, label):
    d = C.load_long(path)
    R, meta = C.run_matrix(d)
    models = sorted(meta.model.unique())
    P, _, _ = C.model_props(d, models)
    Xmod = C.modal(P)
    res = {}
    t_mod = C.fast_item_test(Xmod)
    t_mean = C.fast_item_test(P, stoploss=True)
    tab = t_mod[["item", "k", "expected", "p", "flag"]].rename(columns={"k": "modal_k", "expected": "modal_E", "p": "modal_p", "flag": "modal_flag"})
    tab["mean_sum"], tab["mean_E"], tab["mean_p"], tab["mean_flag"] = t_mean.k, t_mean.expected, t_mean.p, t_mean.flag
    # single runs: NDRAW random draws (one run per model)
    F = np.zeros((NDRAW, 25), bool)
    for k in range(NDRAW):
        Xs = np.array([R[(meta.model == m).values][rng.integers(75 if label == "main" else 25)] for m in models])
        F[k] = C.fast_item_test(Xs).flag.to_numpy()
    tab["single_flag_freq"] = F.mean(0)
    res["observed"] = dict(modal=dict(rate=float(t_mod.flag.mean()), flagged=list(t_mod.item[t_mod.flag])),
                           mean=dict(rate=float(t_mean.flag.mean()), flagged=list(t_mean.item[t_mean.flag])),
                           single=dict(rate_mean=float(F.mean()), rate_q05=float(np.quantile(F.mean(1), .05)),
                                       rate_q95=float(np.quantile(F.mean(1), .95)),
                                       share_draws_any_flag=float((F.sum(1) > 0).mean()),
                                       item_freq={it: float(f) for it, f in zip(C.ITEMS, F.mean(0)) if f >= 0.05}))
    tab.to_csv(os.path.join(C.OUT, f"c2_item_tests_{label}.csv"), index=False)

    # nulls
    TH = np.array([C.theta_ml(Xmod[m][~np.isnan(Xmod[m])], np.where(~np.isnan(Xmod[m]))[0]) for m in range(8)])
    PH = C.p2pl(TH)
    masks = [~np.isnan(R[(meta.model == m).values]) for m in models]
    PQ = np.nanmean(P * (1 - P), 1)
    ratio = PQ / (PH * (1 - PH)).mean(1)
    kappa = np.where(ratio >= 1, np.inf, ratio / np.maximum(1 - ratio, 1e-12))
    nulls = {}
    for kind in ("A0", "B", "C"):
        rates = {"modal": [], "mean": [], "single": []}
        for rep in range(NNULL):
            blocks = []
            for m in range(8):
                M = masks[m]
                if kind == "A0":
                    Y = np.tile((rng.random(25) < PH[m]).astype(float), (M.shape[0], 1))
                elif kind == "B":
                    Y = (rng.random(M.shape) < PH[m]).astype(float)
                else:
                    pm = PH[m] if np.isinf(kappa[m]) else rng.beta(PH[m] * kappa[m], (1 - PH[m]) * kappa[m])
                    Y = (rng.random(M.shape) < pm).astype(float)
                Y[~M] = np.nan
                blocks.append(Y)
            Rn = np.vstack(blocks)
            f = tests(*reduce_all(Rn, meta, models, rng))
            for k2 in rates:
                rates[k2].append(f[k2].mean())
        nulls[kind] = {k2: dict(mean=float(np.mean(v)), q95=float(np.quantile(v, .95)),
                                share_any=float(np.mean(np.array(v) > 0))) for k2, v in rates.items()}
        print(label, kind, {k2: round(np.mean(v), 4) for k2, v in rates.items()}, flush=True)
    res["nulls"] = nulls
    res["setup"] = dict(thetas=TH.round(3).tolist(), kappa=np.round(kappa, 2).tolist())
    return res, tab


if __name__ == "__main__":
    out = {}
    for path, label in ((os.path.join(C.ROOT, "data", "responses_free25.jsonl"), "main"),
                        (os.path.join(C.ROOT, "data", "responses_free25_temp0.jsonl"), "temp0")):
        res, tab = run_arm(path, label)
        out[label] = res
        print(label, json.dumps(res["observed"], indent=1), flush=True)
        print(tab.round(4).to_string(index=False), flush=True)
    C.save_json(out, "c2_reductions.json")
