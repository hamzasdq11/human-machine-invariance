"""Camera-ready tables and inline numbers, generated from results/ (no hand-typed numbers).
Writes paper_cr/tables/*.tex and paper_cr/tables/numbers.tex."""
import os, sys, json, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scripts.sapa_common import ITEMS, load_machine, RES
T = os.path.join(os.path.dirname(__file__), "..", "paper_cr", "tables")
J = lambda f: json.load(open(os.path.join(RES, f)))
C = lambda f: pd.read_csv(os.path.join(RES, f))
from decimal import Decimal, ROUND_HALF_UP
def _r(x, d): return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
f2 = lambda x: _r(x, 2); f3 = lambda x: _r(x, 3)
def sgn(x, d=2): return ("$-$" if x < 0 else "$+$") + _r(abs(x), d)
def ci(x, B=5000, seed=0):
    r = np.random.default_rng(seed); bs = [r.choice(x, len(x)).mean() for _ in range(B)]; return np.percentile(bs, [2.5, 97.5])
N = {}
def put(k, v): N[k] = v

# ---------------- data / calibration
cal = C("sapa_calibration.csv")
put("calN", "95{,}166"); put("calObs", f"{int(cal.N.sum()):,}".replace(",", "{,}"))
put("calNmin", f"{int(cal.N.min()):,}".replace(",", "{,}")); put("calNmax", f"{int(cal.N.max()):,}".replace(",", "{,}"))
put("calAmin", f2(cal.a.min())); put("calAmax", f2(cal.a.max())); put("calBmin", sgn(cal.b.min())); put("calBmax", sgn(cal.b.max()))
rows = "\n".join(f"{r.item} & {'VR' if r.item.startswith('VR') else 'LN'} & {int(r.N):,} & {r.p:.3f} & {r.a:.2f} & {sgn(r.b)} \\\\".replace(",", "{,}") for r in cal.itertuples())
open(f"{T}/tab_calibration.tex", "w").write(r"""\begin{table}[t]
\caption{The 25 text-administrable ICAR items, 2PL calibration on the SAPA release ($N=95{,}166$ respondents, """ + N["calObs"] + r""" observed responses; matrix-sampled, absent responses omitted from the likelihood). VR = verbal reasoning, LN = letter and number series.}
\label{tab:calibration}\centering\small
\begin{tabular}{llrrrr}\toprule
item & dom. & $N$ & $p$ & $\hat a$ & $\hat b$ \\\midrule
""" + rows + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# ---------------- human controls at N = 600 (A6)
a6 = C("a6_matched_controls25.csv")
g = a6[a6.contrast == "gender"]; p = a6[a6.contrast == "placebo"]
glo, ghi = ci(g.rate.values, seed=1); plo, phi = ci(p.rate.values, seed=2)
put("gender", f3(g.rate.mean())); put("genderLo", f3(glo)); put("genderHi", f3(ghi)); put("genderMu", f2(g.mu.mean()))
put("placebo", f3(p.rate.mean())); put("placeboLo", f3(plo)); put("placeboHi", f3(phi))
put("placeboZero", f"{100*(p.rate==0).mean():.0f}"); put("genderZero", f"{100*(g.rate==0).mean():.0f}"); put("ctrlReps", str(len(g)))
open(f"{T}/tab_controls.tex", "w").write(r"""\begin{table}[t]
\caption{Human control quantities in the regime of the anchored machine analysis: focal group of 600 never used in calibration, item parameters fixed, """ + N["ctrlReps"] + r""" replicates per contrast. The placebo recalibrates on the remaining 94{,}566 respondents each time; the gender contrast calibrates once on all women and draws 600 men.}
\label{tab:controls}\centering
\begin{tabular}{lrcr}\toprule
contrast & DIF rate & 95\% CI & no flag \\\midrule
random-split placebo & """ + N["placebo"] + " & [" + N["placeboLo"] + ", " + N["placeboHi"] + "] & " + N["placeboZero"] + r"""\% \\
gender (men vs.\ women) & """ + N["gender"] + " & [" + N["genderLo"] + ", " + N["genderHi"] + "] & " + N["genderZero"] + r"""\% \\
\bottomrule\end{tabular}\end{table}
""")

# ---------------- machine accounting (A8) + person fit (A11)
R, meta, long, _ = load_machine("data/responses_free25.jsonl")
T0 = pd.read_json("data/responses_free25_temp0.jsonl", lines=True)
info = {"Qwen/Qwen2.5-3B-Instruct": ("Qwen2.5-3B", "Qwen", "3B", "16-bit", .350),
        "microsoft/Phi-3.5-mini-instruct": ("Phi-3.5-mini", "Phi", "3.8B", "16-bit", .429),
        "Qwen/Qwen2.5-7B-Instruct": ("Qwen2.5-7B", "Qwen", "7B", "NF4", .421),
        "allenai/OLMo-2-1124-7B-Instruct": ("OLMo-2-7B", "OLMo", "7B", "NF4", .350),
        "teknium/OpenHermes-2.5-Mistral-7B": ("OpenHermes-2.5", "Mistral", "7B", "NF4", .375),
        "NousResearch/Meta-Llama-3.1-8B-Instruct": ("Llama-3.1-8B", "Llama", "8B", "NF4", .331),
        "allenai/OLMo-2-1124-13B-Instruct": ("OLMo-2-13B", "OLMo", "13B", "NF4", .350),
        "Qwen/Qwen2.5-14B-Instruct": ("Qwen2.5-14B", "Qwen", "14B", "NF4", .456)}
pf = C("a11_person_fit25.csv").set_index("model")
pg = {d["model"]: d for d in J("pilot_gate.json")["models"]}
rows = []
for m, (nm, fam, sz, w, _pa) in info.items():
    s = long[long.model == m]; pa = pg[m]["pilot_acc"]; assert pg[m]["admitted"]
    rows.append(f"{nm} & {fam} & {sz} & {w} & {pa:.2f} & {s.parsable.mean():.2f} & {s.y.mean():.2f} & {sgn(pf.loc[m,'theta'])} & {sgn(pf.loc[m,'lz_star'])} \\\\")
open(f"{T}/tab_models.tex", "w").write(r"""\begin{table*}[t]
\caption{The model pool. Fifteen candidates were piloted on all 25 items; seven failed the admission gate (five at floor: SmolLM2-360M, Qwen2.5-0.5B, TinyLlama-1.1B, Qwen2.5-1.5B, SmolLM2-1.7B; two on parse rate: zephyr-7b-beta, phi-4). Each admitted model gave 1{,}875 main-arm and 625 temperature-0 responses; none was excluded (but see D11). Parse and accuracy are main-arm. $\hat\theta$ and $l_z^*$ treat each model as one examinee answering once (its modal answer per item; Section~\ref{sec:unit}). NF4 = 4-bit NormalFloat.}
\label{tab:models}\centering
\begin{tabular}{llllrrrrr}\toprule
model & family & size & weights & pilot acc. & parse & acc. & $\hat\theta$ & $l_z^*$ \\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
rej = [d for d in J('pilot_gate.json')['models'] if not d['admitted']]; put('nCand', str(len(pg))); put('nRej', str(len(rej)))
put("nGen", "15{,}000"); put("nParsed", f"{int(long.parsable.sum()):,}".replace(",", "{,}"))
put("parse", f3(long.parsable.mean())); put("acc", f3(long.y.mean()))
put("parseVfive", f2(long[long.prompt_variant == 'v5'].parsable.mean()))
put("parseVothersLo", f2(long[long.prompt_variant != 'v5'].groupby('prompt_variant').parsable.mean().min()))
put("parseVothersHi", f2(long[long.prompt_variant != 'v5'].groupby('prompt_variant').parsable.mean().max()))
put("tzParse", f3(T0.parsable.mean())); put("tzAcc", f3(T0[T0.parsable].correct.mean()))
put("lzMin", sgn(pf.lz_star.min())); put("lzMax", sgn(pf.lz_star.max()))
put("lzPmin", f2(pf.p_sim.min())); put("lzPmax", f2(pf.p_sim.max()))
put("lzRealMin", f"{100*pf.real_share_humans_worse.min():.0f}"); put("lzRealMax", f"{100*pf.real_share_humans_worse.max():.0f}")
put("thetaMin", sgn(pf.theta.min())); put("thetaMax", sgn(pf.theta.max()))

# ---------------- arms (alpha) A10
a10 = J("a10_alpha25.json"); eight = [ITEMS.index(i) for i in a10["items"]]
put("armFourAcc", f3(np.nanmean(R[:, eight]))); put("armFourAlpha", sgn(a10["machine_alpha_pairwise"], 3))
put("armFourAlphaLw", sgn(a10["machine_alpha_listwise"], 3)); put("armFourLwN", str(a10["machine_listwise_n"]))
put("alphaTwentyFive", f2(a10["machine_alpha_pairwise_25"])); put("alphaHumanTwentyFive", f2(a10["sapa_alpha_pairwise_25"]))
put("alphaHumanEight", f2(a10["sapa_alpha_pairwise_8"]))

# ---------------- clustering (Table IX) + variance components
clall = C("clustering25_main.csv"); cl = clall.head(8); cj = J("clustering25_main.json")
rows = "\n".join(f"{r.item} & {r.p:.2f} & {r.icc:.2f} & {r.deff:.1f} & {r.neff:.0f} & {r.pmin:.2f}--{r.pmax:.2f} \\\\" for r in cl.itertuples())
open(f"{T}/tab_cluster.tex", "w").write(r"""\begin{table}[t]
\caption{Clustering of the run-level machine group by model: the eight items with the largest design effects. $p$ pooled proportion correct, ICC intraclass correlation across models, deff the design effect, $N_{\text{eff}}=n/\text{deff}$ the effective sample size, $n$ being the item's scored responses (""" + f"{clall.n.min():.0f}--{clall.n.max():.0f} across the 25 items, median {cj['n_median']:.0f}" + r"""). Last column: range of per-model proportions.}
\label{tab:cluster}\centering
\begin{tabular}{lrrrrc}\toprule
item & $p$ & ICC & deff & $N_{\text{eff}}$ & per-model $p$ \\\midrule
""" + rows + "\n\\bottomrule\\end{tabular}\\end{table}\n")
put("deffMed", f"{cj['deff_median']:.1f}"); put("deffMax", f"{cj['deff_max']:.1f}"); put("neffMed", f"{cj['neff_median']:.0f}")
put("iccMed", f2(cj["icc_median"])); put("iccMax", f2(cj["icc_max"])); put("seInfl", f"{cj['se_inflation']:.1f}")
put("nItemMed", f"{cj['n_median']:.0f}")
put("vcModel", f3(cj["varcomp"]["model"])); put("vcPrompt", f3(cj["varcomp"]["prompt_variant"])); put("vcSeed", f3(cj["varcomp"]["seed"]))
put("sdObs", f"{cj['score_sd_observed']:.4f}"); put("sdBin", f"{cj['score_sd_binomial']:.4f}"); put("cellRatio", f2(cj["within_cell_var_ratio"]))
put("sdPB", f"{cj['score_sd_poisson_binomial']:.4f}"); put("sdPBmod", f"{cj['score_sd_poisson_binomial_models']:.4f}")  # Tier 0: replaces sdBin in the text
put("neffAtMed", f"{cj['neff_at_median_deff']:.0f}")  # Tier 0: n_median / deff_median, consistent with seInfl = sqrt(deff_median)
_word = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
put("frameSig", _word[cj["n_models_prompt_bh"]]); put("seedSig", _word[cj["n_models_seed_bh"]])  # Tier 0: within-model F, BH over models
put("frameMaxModel", info[cj["frame_range_max"]["model"]][0]); put("frameMaxLo", f2(cj["frame_range_max"]["lo"])); put("frameMaxHi", f2(cj["frame_range_max"]["hi"]))
_ratio = clall.set_index("item").n / C("sapa_calibration.csv").set_index("item").N  # Tier 0: focal/reference ratio per item (matrix-sampled reference)
put("ratioLo", f3(_ratio.min())); put("ratioHi", f3(_ratio.max()))

# ---------------- run-level DIF vs replicated-human null (A13)
inv = J("inv25_main_summary.json"); a13 = C("a13_replicated_human_null25.csv"); a1 = C("a1_model_fit25_main.csv")
put("mhBC", f2(inv["ets_BC_rate"]))  # Tier 0: registered MH estimator, run level only (D13)
put("runRate", f2(inv["dif_rate_total"])); put("runFlag", str(inv["n_flagged"])); put("runMu", sgn(inv["focal_mu"]))
put("nullRunMean", f2(a13.rate.mean())); put("nullRunLo", f2(a13.rate.quantile(.05))); put("nullRunHi", f2(a13.rate.quantile(.95)))
put("nullRunP", f2((a13.rate >= inv["dif_rate_total"] - 1e-9).mean())); put("nullSigZero", f"{100*(a13.sigma < 1e-3).mean():.0f}")
mods = a1[a1.kind == "model"]; hum = a1[a1.kind == "human_random"]
obs_rmsd = mods.ms_rmsd.median()
put("aoneRmsdObs", f3(obs_rmsd)); put("aoneRmsdNull", f3(a13.rmsd_med.median())); put("aoneRmsdNullHi", f3(a13.rmsd_med.quantile(.95)))
put("aoneRmsdP", f3((a13.rmsd_med >= obs_rmsd).mean())); put("aoneHumRmsdMax", f3(a1[a1.kind != 'model'].ms_rmsd.max()))
put("aoneRmsdLo", f2(mods.ms_rmsd.min())); put("aoneRmsdHi", f2(mods.ms_rmsd.max()))
put("nullReps", str(len(a13)))
eb = J("exact_bootstrap25_main.json"); put("ebLo", f2(eb["ci95"][0])); put("ebHi", f2(eb["ci95"][1]))
ra = C("a3_random_anchors25_main.csv"); put("raLo", f2(ra.rate.min())); put("raHi", f2(ra.rate.max()))
open(f"{T}/tab_runlevel.tex", "w").write(r"""\begin{table}[t]
\caption{Run-level statistics against the replicated-respondent null: eight synthetic humans, invariant by construction, each answers once at one model's $\hat\theta$ and is re-run as that model was (its 75 runs, its missing cells, its own run-to-run flip rate). """ + N["nullReps"] + r""" null data sets; the published pipeline is applied unchanged. $^\dagger$Median over the 8 models of the misfit (RMSD) of run-level item proportions to the best-fitting human ability distribution.}
\label{tab:runlevel}\centering\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}\toprule
statistic & obs. & null med.\ [90\%] & $P(\ge\text{obs})$ \\\midrule
anchored DIF rate & """ + N["runRate"] + " & " + f2(a13.rate.median()) + " [" + N["nullRunLo"] + ", " + N["nullRunHi"] + "] & " + N["nullRunP"] + r""" \\
model-level RMSD$^\dagger$ & """ + N["aoneRmsdObs"] + " & " + N["aoneRmsdNull"] + " [" + f3(a13.rmsd_med.quantile(.05)) + ", " + N["aoneRmsdNullHi"] + "] & " + N["aoneRmsdP"] + r""" \\
\bottomrule
\end{tabular}\end{table}
""")

# ---------------- model-as-unit item test (A12)
a12 = C("a12_eight_examinees25.csv"); a12j = J("a12_eight_examinees25.json"); var = J("a12_variants25.json")
specs = [v["flagged"] for v in var.values()] + [a12j["observed_flagged"], a12j["temp0_single_flagged"]]
cnt = collections.Counter(i for s in specs for i in s); rates = [v["rate"] for v in var.values()] + [a12j["observed_rate"], a12j["temp0_single_rate"]]
put("nSpecs", str(len(specs))); put("specLo", f2(min(rates))); put("specMed", f2(np.median(rates))); put("specHi", f2(max(rates)))
put("unitRate", f2(a12j["observed_rate"])); put("unitFlag", str(int(a12.flag.sum())))
put("realEightMean", f3(a12j["real_random8_mean"])); put("realEightMax", f2(a12j["real_random8_max"])); put("realEightQ", f2(a12j["real_random8_q95"]))
put("menEightMean", f3(a12j["real_men8_mean"])); put("menEightMax", f2(a12j["real_men8_max"]))
put("simEightMean", f3(a12j["sim_null_mean"])); put("unitP", f3(a12j["p_observed_vs_real_random"]))
put("tzUnitRate", f2(a12j["temp0_single_rate"]))
put("cntVRthirtysix", str(cnt["VR.36"])); put("cntVRthirtynine", str(cnt["VR.39"]))
fam = [v["rate"] for k, v in var.items() if k.startswith("family_")]; put("famLo", f2(min(fam))); put("famHi", f2(max(fam)))
loo = [v["rate"] for k, v in var.items() if k.startswith("drop_")]; put("looLo", f2(min(loo))); put("looHi", f2(max(loo)))
pr = [v["rate"] for k, v in var.items() if k.startswith("prompt_")]; put("prLo", f2(min(pr))); put("prHi", f2(max(pr)))
put("guessUnit", f2(var["guess_floor_c125"]["rate"]))
calb = cal.set_index("item")
rows = []
for r in a12.itertuples():
    pv = "$<$0.001" if r.p < 0.001 else f"{r.p:.3f}"
    it = f"\\textbf{{{r.item}}}" if r.flag else r.item
    pw3 = J("a12_power25.json")["3.0"]["power_by_item"][r.item]
    rows.append(f"{it} & {sgn(calb.loc[r.item,'b'])} & {r.k}/{r.n} & {r.expected:.1f} & {pv} & {f2(pw3)} & {cnt.get(r.item,0)} \\\\")
open(f"{T}/tab_unit.tex", "w").write(r"""\begin{table}[t]
\caption{Item-level test with the model as the examinee ($N=8$, one modal answer each). $k/n$: models answering correctly; E: expected count under invariance, $\sum_m P_i(\hat\theta_m)$ with $\hat\theta_m$ from the model's other items; $p$: exact two-sided Poisson-binomial. Bold: flagged at BH $0.05$. pow.: simulated power of this test for a 3-logit shift making the item easier for the models (Table~\ref{tab:power}). Last column: specifications (of """ + N["nSpecs"] + r""") in which the item is flagged.}
\label{tab:unit}\centering\small
\setlength{\tabcolsep}{4pt}\begin{tabular}{lrrrrrr}\toprule
item & $b_{\text{ref}}$ & $k/n$ & E & $p$ & pow. & specs \\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# ---------------- power with 8 examinees (Tier 1: both directions, same design)
pw = J("a12_power25.json")
def _pwrow(d, v):
    h = v.get("harder")
    hc = f" & {f2(h['median_power'])} & {h['items_ge_08']}" if h else ""
    return f"{float(d):.1f} & {f2(v['median_power'])} & {v['items_ge_08']}{hc} & {f3(v['typeI'])} \\\\"
rows = "\n".join(_pwrow(d, v) for d, v in pw.items())
open(f"{T}/tab_power.tex", "w").write(r"""\begin{table}[t]
\caption{Power of the model-as-examinee test, from 8 simulated 2PL examinees at the models' $\hat\theta$, five items at a time made easier or harder by $d$ logits for all eight (40 data sets per item, shift and direction). Median: median per-item power; $\ge 0.8$: items with power of at least 0.8; Type-I: flag rate on the unshifted items when the shifted ones are easier.}
\label{tab:power}\centering\setlength{\tabcolsep}{4pt}
\begin{tabular}{rrrrrr}\toprule
 & \multicolumn{2}{c}{easier for models} & \multicolumn{2}{c}{harder for models} & \\\cmidrule(lr){2-3}\cmidrule(lr){4-5}
shift $d$ & median & $\ge 0.8$ & median & $\ge 0.8$ & Type-I \\\midrule
""" + rows + "\n\\bottomrule\\end{tabular}\\end{table}\n")
put("powTwo", f2(pw["2.0"]["median_power"])); put("powThree", f2(pw["3.0"]["median_power"])); put("powOne", f2(pw["1.0"]["median_power"]))
# Tier 1: the harder direction, and the items it can still reach at three logits
_h3 = pw["3.0"]["harder"]
put("powHarderThree", f2(_h3["median_power"]))
_top = sorted(_h3["power_by_item"].items(), key=lambda kv: -kv[1])
_best = [k for k, v in _top if v == _top[0][1]]
put("powHarderTop", " and ".join(sorted(_best))); put("powHarderTopPow", f2(_top[0][1]))

# ---------------- robustness of the run-level number
rob = [json.loads(l) for l in open(os.path.join(RES, "robustness25.jsonl"))]
rr = {o["tag"]: o["dif_rate"] for o in rob}
put("robWrong", f2(rr["unparsable_as_wrong"])); put("robLenient", f2(rr["lenient_parser"])); put("robNoVfive", f2(rr["drop_v5"]))
lopo = [v for k, v in rr.items() if k.startswith("drop_prompt")]; put("robLopoLo", f2(min(lopo))); put("robLopoHi", f2(max(lopo)))
rb = J("robust_refboot25.json"); put("refbootLo", f2(rb["dif_rate_min"])); put("refbootHi", f2(rb["dif_rate_max"]))
put("guessRun", f2(J("robust_guess25.json")["dif_rate"])); put("tzRun", f2(J("inv25_temp0_summary.json")["dif_rate_total"]))
au = J("audit_scoring25.json"); put("auditRows", "20{,}000")

with open(f"{T}/numbers.tex", "w") as f:
    for k, v in N.items(): f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
print(len(N), "numbers;", sorted(os.listdir(T)))

# ---------------- administration formats (arms 1-4), alpha on the 8 items with item-level human data
g1 = J("generation_arm_diagnostics.json"); lp = J("logprob_arm_diagnostics.json")["scoring_rules"]
arms = [("1. numbered choice", g1["icar_accuracy"], g1["alpha_machine"]),
        ("2. log-prob, 8 options", lp["mean_logprob_all8"]["acc"], lp["mean_logprob_all8"]["alpha"]),
        ("3. log-prob, 6 options", lp["mean_logprob_6subst"]["acc"], lp["mean_logprob_6subst"]["alpha"]),
        ("4. free response + gate", float(np.nanmean(R[:, eight])), a10["machine_alpha_pairwise"])]
rows = "\n".join(f"{n} & {a:.3f} & {sgn(al,3)} \\\\" for n, a, al in arms)
open(f"{T}/tab_arms.tex", "w").write(r"""\begin{table}[t]
\caption{Four administration formats, scored on the eight items with item-level human data. Coefficient $\alpha$ across machine respondents; the human value on the same items is $+0.766$ (psychTools) and $+""" + N["alphaHumanEight"] + r"""$ (SAPA). Arms 1--3 used the earlier pool (Section~\ref{sec:formats}); arm 4 is the analysed administration.}
\label{tab:arms}\centering
\begin{tabular}{lrr}\toprule
arm & acc. & $\alpha$ \\\midrule
""" + rows + "\n\\bottomrule\\end{tabular}\\end{table}\n")
print("arms table written")
