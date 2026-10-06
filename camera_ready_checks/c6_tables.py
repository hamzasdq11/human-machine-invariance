"""C6: LaTeX fragments for the camera-ready, generated from the c1-c4 results files.

tables/ck_numbers.tex   macros (prefix \\ck...) for every number quoted in paper_additions.tex
tables/tab_reductions.tex, tab_bundle.tex, tab_runlevel_v2.tex, tab_power_v2.tex
Tier 2: item-property macros (\\ckIp...) from c9_item_properties.json; letter-series bundle robustness macros
(\\ckBx...) and tab_bundle.tex from c11_bundle_anchor.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import common as C

T = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(T, exist_ok=True)
J = lambda f: json.load(open(os.path.join(C.OUT, f)))
c1, c2, c3, c4 = J("c1_verify.json"), J("c2_reductions.json"), J("c3_bundles.json"), J("c4_runlevel_nulls.json")
hc = pd.read_csv(os.path.join(C.ROOT, "results", "a6_matched_controls25.csv"))
n4 = pd.read_csv(os.path.join(C.OUT, "c4_runlevel_nulls.csv"))
pw = pd.read_csv(os.path.join(C.OUT, "c1_power_one_item.csv"))
c7 = J("c7_humans.json") if os.path.exists(os.path.join(C.OUT, "c7_humans.json")) else None
# Tier 1: the paper's own replicated-human null (one answer + flips), so each null has one source in the paper
a13 = pd.read_csv(os.path.join(C.ROOT, "results", "a13_replicated_human_null25.csv"))
c8 = J("c8_purified.json") if os.path.exists(os.path.join(C.OUT, "c8_purified.json")) else None

from decimal import Decimal, ROUND_HALF_UP
_r = lambda x, d: str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
f2 = lambda v: _r(v, 2)
f3 = lambda v: _r(v, 3)
def fp(v, floor):
    return f"$<${floor}" if v <= float(floor) + 1e-12 else (f"{v:.3f}" if v < 0.1 else f"{v:.2f}")
items = lambda L: ", ".join(L) if L else "none"

M = {}
# ---- run-level nulls
for k, lab in (("A", "NullA"), ("B", "NullB"), ("C", "NullC")):
    g = n4[n4.kind == k].rate
    M[f"ck{lab}Mean"] = f2(g.mean()); M[f"ck{lab}Lo"] = f2(g.quantile(.05)); M[f"ck{lab}Hi"] = f2(g.quantile(.95))
    M[f"ck{lab}Ge"] = f2((g >= 0.84 - 1e-9).mean())
M["ckNullReps"] = str(int(n4.groupby("kind").size().min()))
# ---- reductions
o, z = c2["main"]["observed"], c2["temp0"]["observed"]
M["ckModalRate"], M["ckMeanRate"] = f2(o["modal"]["rate"]), f2(o["mean"]["rate"])
M["ckSingleRate"], M["ckSingleLo"], M["ckSingleHi"] = f2(o["single"]["rate_mean"]), f2(o["single"]["rate_q05"]), f2(o["single"]["rate_q95"])
M["ckSingleVRthirtysix"] = f"{100*o['single']['item_freq'].get('VR.36',0):.0f}"
M["ckSingleVRthirtynine"] = f"{100*o['single']['item_freq'].get('VR.39',0):.0f}"
M["ckSingleVRtwentysix"] = f"{100*o['single']['item_freq'].get('VR.26',0):.0f}"
M["ckTzModalRate"], M["ckTzMeanRate"], M["ckTzSingleRate"] = f2(z["modal"]["rate"]), f2(z["mean"]["rate"]), f2(z["single"]["rate_mean"])
nl = c2["main"]["nulls"]
for k in ("A0", "B", "C"):
    for red in ("modal", "mean", "single"):
        M[f"ckRedNull{k.replace('0','zero')}{red.capitalize()}"] = f3(nl[k][red]["mean"])
# ---- bundles
b, bz = c3["main"], c3["temp0"]
L = b["letter_given_VR"]
M["ckLetModalObs"], M["ckLetModalExp"] = f"{L['modal']['total']:.0f}", f"{L['modal']['expected']:.1f}"
M["ckLetMeanObs"], M["ckLetMeanExp"] = f"{L['mean']['total']:.1f}", f"{L['mean']['expected']:.1f}"
cal = b["calibration"]
nmin_main = 1 / (int(os.environ.get("NNULL", 1500)) + 1)
WORD = {"0.9": "Nine", "0.8": "Eight", "0.7": "Seven", "0.6": "Six", "0.5": "Five"}
def kkey(kind):
    if kind == "A0":
        return "Azero"
    if "_rho" in kind:
        return "CRho" + WORD[kind.split("_rho")[1]]
    return kind
for kind in ("A0", "B", "C", "C_rho0.8", "C_rho0.7", "C_rho0.6", "C_rho0.5"):
    key = kkey(kind)
    M[f"ckLetCal{key}"] = fp(cal[kind]["mean:letter_given_VR"]["calibrated_p"], "0.001")
    M[f"ckLNCal{key}"] = fp(cal[kind]["mean:LN_given_VR"]["calibrated_p"], "0.001")
calz = bz["calibration"]
for kind in ("A0", "B", "C", "C_rho0.8", "C_rho0.5"):
    key = kkey(kind)
    M[f"ckTzLetCal{key}"] = fp(calz[kind]["mean:letter_given_VR"]["calibrated_p"], "0.003")
Lz = bz["letter_given_VR"]
M["ckTzLetMeanObs"], M["ckTzLetMeanExp"] = f"{Lz['mean']['total']:.1f}", f"{Lz['mean']['expected']:.1f}"
pm = pd.DataFrame(b["per_model"])
below = (pm.letter_mean < pm.letter_expected_from_VR - 0.02).to_numpy()   # Tier 2: ranges over the models below the line
M["ckLetAccLo"], M["ckLetAccHi"] = f2(pm.letter_mean[below].min()), f2(pm.letter_mean[below].max())
M["ckLetExpLo"], M["ckLetExpHi"] = f2(pm.letter_expected_from_VR[below].min()), f2(pm.letter_expected_from_VR[below].max())
M["ckLetBelow"] = str(int(below.sum()))
assert (~below).sum() == 1 and pm.model[~below].iloc[0] == "Meta-Llama-3.1-8B"
M["ckLetOnLine"] = f2(pm.letter_mean[~below].iloc[0])
pwr = c3["main"]["power_letter_shift"]
M["ckBunPowHalf"], M["ckBunPowSeventyFive"], M["ckBunPowOne"] = f2(pwr["0.5"]["bundle"]), f2(pwr["0.75"]["bundle"]), f2(pwr["1.0"]["bundle"])
M["ckBunItemPowTwo"] = f2(pwr["2.0"]["any_item"])
LN = b["LN_given_VR"]
M["ckLNMeanObs"], M["ckLNMeanExp"] = f"{LN['mean']['total']:.1f}", f"{LN['mean']['expected']:.1f}"
# ---- power, one item at a time
for d in (1.0, 2.0, 3.0, 4.0):
    for dr in ("easier", "harder"):
        g = pw[(pw["shift"] == d) & (pw.direction == dr)].power
        M[f"ckPow{dr.capitalize()}{['','One','Two','Three','Four'][int(d)]}"] = f2(g.median())
        M[f"ckPowN{dr.capitalize()}{['','One','Two','Three','Four'][int(d)]}"] = str(int((g >= .8).sum()))
M["ckSimNull"] = f3(c1["sim_null"]["mean"])
if c7 is not None:
    g5, g8 = c7["real_groups_of_eight"]["ge5_other"], c7["real_groups_of_eight"]["ge8_other"]
    M["ckHumPbMean"], M["ckHumPbQ"], M["ckHumPbMax"] = f3(g5["exact_pb"]["mean"]), f2(g5["exact_pb"]["q95"]), f2(g5["exact_pb"]["max"])
    M["ckHumSlMean"], M["ckHumSlQ"], M["ckHumSlMax"] = f3(g5["stoploss"]["mean"]), f2(g5["stoploss"]["q95"]), f2(g5["stoploss"]["max"])
    pnum = lambda v: f"{max(v, 0.001):.3f}"
    M["ckHumPModal"] = pnum(g5["exact_pb"]["p_ge_modal_020"])
    M["ckHumPSingle"] = pnum(g5["exact_pb"]["p_ge_single_012"])
    M["ckHumPRunMean"] = pnum(g5["stoploss"]["p_ge_runmean_008"])
    M["ckHumEightPRunMean"] = pnum(g8["stoploss"]["p_ge_runmean_008"])
    M["ckHumEightPbMean"], M["ckHumEightSlMean"] = f3(g8["exact_pb"]["mean"]), f3(g8["stoploss"]["mean"])
    M["ckHumReps"] = "1{,}000"
    M["ckCalDiff"] = f"{np.ceil(1000 * max(c7['calibration']['max_abs_diff_a'], c7['calibration']['max_abs_diff_b'])) / 1000:.3f}"
    rv = c7["rho"]["VR_letter"]
    M["ckRhoVL"], M["ckRhoVLLo"], M["ckRhoVLHi"] = f2(rv["rho"]), f2(rv["ci95"][0]), f2(rv["ci95"][1])
    M["ckRhoVLN"] = f"{rv['persons_with_both']:,}".replace(",", "{,}")
    M["ckRhoVLNall"] = f2(c7["rho"]["VR_LN"]["rho"]); M["ckRhoVNum"] = f2(c7["rho"]["VR_number"]["rho"])
    lb = c7.get("letter_bundle_at_rho", {})
    if lb:
        M["ckLetCalRhoHat"] = fp(lb["main_rho"]["calibrated_p_mean"], "0.001")
        M["ckLetCalRhoLow"] = fp(lb["main_ci_low"]["calibrated_p_mean"], "0.001")
        M["ckLetModalCalRhoHat"] = fp(lb["main_rho"]["calibrated_p_modal"], "0.001")
        M["ckTzLetCalRhoHat"] = fp(lb["temp0_rho"]["calibrated_p_mean"], "0.001")
        M["ckTzLetModalCalRhoHat"] = fp(lb["temp0_rho"]["calibrated_p_modal"], "0.001")
    lnb = c7.get("ln_bundle_at_rho", {})
    if lnb:
        M["ckLNCalRhoHat"] = f2(lnb["main"]["calibrated_p"]); M["ckTzLNCalRhoHat"] = f2(lnb["temp0"]["calibrated_p"])

if c8 is not None:
    WORD = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
            9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
    combos = {"VR.36": 0, "VR.39": 0, "VR.26": 0}; ncomb = 0
    for arm in ("main", "temp0"):
        for red in ("modal", "mean"):
            for key in ("start", "final"):
                ncomb += 1
                for it in combos: combos[it] += it in c8[arm][red][key]
        for key in ("start_share", "final_share"):
            ncomb += 1
            for it in combos: combos[it] += c8[arm]["single"][key][it] > 0.5
    M["ckPurCombs"] = WORD[ncomb]
    M["ckPurVRthirtysix"], M["ckPurVRthirtynine"], M["ckPurVRtwentysix"] = WORD[combos["VR.36"]], WORD[combos["VR.39"]], WORD[combos["VR.26"]]
    M["ckPurModalMainN"] = WORD[len(c8["main"]["modal"]["final"])]
    M["ckPurModalMainAtBound"] = WORD[c8["main"]["modal"]["models_at_bound"]]
    M["ckPurModalMainLeft"] = {16: "sixteen", 15: "fifteen", 17: "seventeen", 18: "eighteen", 14: "fourteen"}.get(c8["main"]["modal"]["items_left"], str(c8["main"]["modal"]["items_left"]))
    M["ckPurSingleVRthirtynine"] = f"{100 * c8['main']['single']['final_share']['VR.39']:.0f}"

# ---- item properties (c9, Tier 2): Spearman correlation of the implied logit shift with item properties, main arm
c9 = J("c9_item_properties.json") if os.path.exists(os.path.join(C.OUT, "c9_item_properties.json")) else None
if c9 is not None:
    sg = lambda v: ("$-$" if float(_r(v, 2)) < 0 else "") + _r(abs(v), 2)
    ip = c9["main"]["all25"]
    for prop, key in (("a", "A"), ("b", "B"), ("words", "Words")):
        M[f"ckIp{key}"], M[f"ckIp{key}Lo"], M[f"ckIp{key}Hi"] = sg(ip[prop]["rho"]), sg(ip[prop]["ci95"][0]), sg(ip[prop]["ci95"][1])
    M["ckIpWordsNoAnt"] = sg(c9["main"]["all25_excl_antonyms"]["words"]["rho"])

# ---- letter-series bundle robustness grid (c11, Tier 2): run dependence x dimensionality x verbal anchor
c11 = J("c11_bundle_anchor.json") if os.path.exists(os.path.join(C.OUT, "c11_bundle_anchor.json")) else None
if c11 is not None:
    g = lambda arm, anc, test, cell: c11[arm][anc][test]["calibrated"][cell]["p"]
    one_d = ("1D_fixed", "1D_fresh", "1D_matched")
    M["ckBxOneDMax"] = fp(max(g("main", "VR16", "letter_mean", c) for c in one_d), "0.001")
    M["ckBxTwoDMatched"] = fp(g("main", "VR16", "letter_mean", "2D_matched"), "0.001")
    M["ckBxTwoDFixed"] = fp(g("main", "VR16", "letter_mean", "2D_fixed"), "0.001")
    M["ckTzBxTwoDMatched"] = fp(g("temp0", "VR16", "letter_mean", "2D_matched"), "0.001")
    M["ckTzBxTwoDFixed"] = fp(g("temp0", "VR16", "letter_mean", "2D_fixed"), "0.001")
    M["ckBxPurTwoExp"] = f"{c11['main']['VRminus2']['letter_mean']['expected']:.1f}"
    M["ckBxPurFourExp"] = f"{c11['main']['VRminus4']['letter_mean']['expected']:.1f}"
    allc = one_d + ("2D_fixed", "2D_fresh", "2D_matched")
    M["ckBxPurMax"] = fp(max(g(arm, a, "letter_mean", c) for arm in ("main", "temp0") for a in ("VRminus2", "VRminus4") for c in allc), "0.001")
    M["ckBxMax"] = fp(max(g(arm, a, "letter_mean", c) for arm in ("main", "temp0") for a in ("VR16", "VRminus2", "VRminus4") for c in allc), "0.001")
    M["ckBxLNTwoDMatched"] = f2(g("main", "VR16", "LN_mean", "2D_matched"))
    M["ckBxReps"] = f"{c11['reps_main']:,}".replace(",", "{,}")
    M["ckBxRepsTz"] = f"{c11['reps_temp0']:,}".replace(",", "{,}")

import re as _re
bad = [k for k in M if not _re.fullmatch(r"[A-Za-z]+", k)]
assert not bad, f"macro names with non-letters: {bad}"
with open(os.path.join(T, "ck_numbers.tex"), "w") as f:
    f.write("% generated by camera_ready_checks/c6_tables.py -- do not edit by hand\n")
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---- Table: reductions
fl = lambda L: ", ".join(L) if L else "none"
hum = c7 is not None
hcols = "cc" if hum else ""
hhead = " & \\multicolumn{2}{c}{real groups of 8}" if hum else ""
hsub = " & mean & $p$" if hum else ""
hcm = "\\cmidrule(lr){7-8}" if hum else ""
def hrow(kind):
    if not hum: return ""
    if kind == "modal": return f" & {M['ckHumPbMean']} & $\\le${M['ckHumPModal']}"
    if kind == "single": return f" & {M['ckHumPbMean']} & {M['ckHumPSingle']}"
    return f" & {M['ckHumSlMean']} & {M['ckHumPRunMean']}"
ncol = 8 if hum else 6
humcap = (f" Real groups of 8: {M['ckHumReps']} groups of eight SAPA respondents per item, each answering once (so the three reductions coincide), tested in the same way as the models (run mean: with the stop-loss bound); $p=(1+k)/1{{,}}001$, with $k$ the number of groups at or above the models' rate." if hum else "")
purnote = ""
if c8 is not None:
    pm_, pz_ = c8["main"], c8["temp0"]
    sh = lambda a, it: f"{100*a['single']['final_share'][it]:.0f}"
    purnote = (f" With flagged items excluded from every ability estimate (iterated to a fixed point; single runs: {pm_['single']['draws']} draws): "
               f"main arm, modal {fl(pm_['modal']['final'])} ({M['ckPurModalMainAtBound']} models then answer none of the {M['ckPurModalMainLeft']} items left); "
               f"run mean {fl(pm_['mean']['final'])}; single run VR.36 in {sh(pm_, 'VR.36')}\\%, VR.39 in {sh(pm_, 'VR.39')}\\%, VR.26 in {sh(pm_, 'VR.26')}\\% of draws. "
               f"Temperature-0 arm: modal {fl(pz_['modal']['final'])}; run mean {fl(pz_['mean']['final'])}; single run VR.36 in {sh(pz_, 'VR.36')}\\%, VR.39 in {sh(pz_, 'VR.39')}\\%, VR.26 in {sh(pz_, 'VR.26')}\\%.")
r = f"""\\begin{{table}}[t]
\\caption{{The model-level item test under three ways of reducing a model's runs to one examinee: its modal answer per item (correct when more than half its parsed runs are); one randomly chosen (frame, seed) run, over 2{{,}}000 draws (main arm: mean rate {M['ckSingleRate']}, 90\\% range {M['ckSingleLo']}--{M['ckSingleHi']}; an item counts as flagged when it is flagged in most draws); and its proportion correct per item, tested with the stop-loss bound~(\\ref{{eq:stoploss}}). Null rates: share of items flagged when the eight models are replaced by synthetic humans, invariant by construction, whose runs copy one answer (fixed), are fresh independent draws (fresh), or are draws from fixed propensities matched to each model's run-to-run variance (matched); 400 data sets each.{humcap}}}
\\label{{tab:reductions}}\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabular}}{{lccccc{hcols}}}\\toprule
 & \\multicolumn{{2}}{{c}}{{observed}} & \\multicolumn{{3}}{{c}}{{null rate}}{hhead} \\\\\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-6}}{hcm}
reduction & main & temp.\\ 0 & fixed & fresh & matched{hsub} \\\\\\midrule
modal & {M['ckModalRate']} & {M['ckTzModalRate']} & {M['ckRedNullAzeroModal']} & {M['ckRedNullBModal']} & {M['ckRedNullCModal']}{hrow('modal')} \\\\
single run & {M['ckSingleRate']} & {M['ckTzSingleRate']} & {M['ckRedNullAzeroSingle']} & {M['ckRedNullBSingle']} & {M['ckRedNullCSingle']}{hrow('single')} \\\\
run mean & {M['ckMeanRate']} & {M['ckTzMeanRate']} & {M['ckRedNullAzeroMean']} & {M['ckRedNullBMean']} & {M['ckRedNullCMean']}{hrow('mean')} \\\\\\midrule
\\multicolumn{{{ncol}}}{{p{{0.93\\columnwidth}}}}{{Flagged, main arm. Modal: {fl(o['modal']['flagged'])}. Run mean: {fl(o['mean']['flagged'])}. Single run: VR.36 in {M['ckSingleVRthirtysix']}\\%, VR.39 in {M['ckSingleVRthirtynine']}\\%, VR.26 in {M['ckSingleVRtwentysix']}\\% of draws. Temperature-0 arm. Modal: {fl(z['modal']['flagged'])}. Run mean: {fl(z['mean']['flagged'])}. Single run: VR.36 in {100*z['single']['item_freq'].get('VR.36',0):.0f}\\%, VR.39 in {100*z['single']['item_freq'].get('VR.39',0):.0f}\\%, VR.26 in {100*z['single']['item_freq'].get('VR.26',0):.0f}\\% of draws.{purnote}}} \\\\
\\bottomrule\\end{{tabular}}\\end{{table}}
"""
open(os.path.join(T, "tab_reductions.tex"), "w").write(r)

# ---- Table: bundle (Tier 2: from the c11 robustness grid when it exists; otherwise the c3/c7 version)
if c11 is not None:
    cells = ("1D_fixed", "1D_fresh", "1D_matched", "2D_fixed", "2D_matched")
    def brow(lab, arm, anc, test):
        r = c11[arm][anc][test]
        obs = f"{r['observed']:.0f}" if test.endswith("modal") else f"{r['observed']:.1f}"
        return f"{lab} & {obs} & {r['expected']:.1f} & " + " & ".join(fp(r["calibrated"][c]["p"], "0.001") for c in cells) + " \\\\"
    bt = "\n".join([
        "\\begin{table}[t]",
        "\\caption{Exploratory bundle test, chosen after inspecting Table~\\ref{tab:unit}: the seven letter-series items (LN.05--LN.58), with each model's ability estimated from its verbal items. "
        "Totals are summed over models and items (56 model-item pairs; 72 for LN); mean: each model's proportion correct, stop-loss test; modal: exact test. "
        f"Calibrated $p=(1+k)/(n+1)$, where $k$ of $n$ null data sets give a bundle $p$-value at least as small as the observed one ($n={M['ckBxReps']}$; temperature~0: {M['ckBxRepsTz']}). "
        "Runs are dependent as in the fixed, fresh and matched nulls of Table~\\ref{tab:reductions}. "
        f"1-D: letter-series ability equals verbal ability; 2-D: centred on verbal ability, it departs from it as much as two human abilities correlated $\\hat\\rho={M['ckRhoVL']}$ do ({M['ckRhoVLNall']} for the LN row), the SAPA latent correlations. "
        "Anchor $-2$: verbal ability without VR.36 and VR.39; $-4$: also without VR.26 and VR.42. No $p$-value allows for the choice of the bundle.}",
        "\\label{tab:bundle}\\centering\\footnotesize\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{lrrccccc}\\toprule",
        " & & & \\multicolumn{3}{c}{1-D null} & \\multicolumn{2}{c}{2-D null} \\\\\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}",
        "bundle & obs. & exp. & fixed & fresh & matched & fixed & matched \\\\\\midrule",
        brow("letter, modal", "main", "VR16", "letter_modal"),
        brow("letter, mean", "main", "VR16", "letter_mean"),
        brow("\\quad temp.\\ 0", "temp0", "VR16", "letter_mean"),
        brow("\\quad anchor $-2$", "main", "VRminus2", "letter_mean"),
        brow("\\quad anchor $-4$", "main", "VRminus4", "letter_mean"),
        brow("LN (all 9), mean", "main", "VR16", "LN_mean"),
        "\\bottomrule\\end{tabular}\\end{table}", ""])
    open(os.path.join(T, "tab_bundle.tex"), "w").write(bt)
else:
    lb = (c7 or {}).get("letter_bundle_at_rho", {})
    lnb = (c7 or {}).get("ln_bundle_at_rho", {})
    hatcol = bool(lb and lnb)
    cell = lambda v, fl_="0.001": fp(v, fl_)
    if hatcol:
        hat = dict(modal=cell(lb["main_rho"]["calibrated_p_modal"]), mean=cell(lb["main_rho"]["calibrated_p_mean"]),
                   tz=cell(lb["temp0_rho"]["calibrated_p_mean"]), ln=f2(lnb["main"]["calibrated_p"]))
        hathead, hatnote = "$\\hat\\rho$", f" Column $\\hat\\rho$ uses the latent correlation SAPA respondents show between verbal ability and the bundle's ability: {M['ckRhoVL']} for letter series, {M['ckRhoVLNall']} for the whole LN subscale."
    else:
        hat = dict(modal=fp(cal['C_rho0.8']['modal:letter_given_VR']['calibrated_p'],'0.001'), mean=M['ckLetCalCRhoEight'],
                   tz=M['ckTzLetCalCRhoEight'], ln=M['ckLNCalCRhoEight'])
        hathead, hatnote = "0.8", ""
    bt = f"""\\begin{{table}}[t]
    \\caption{{Exploratory bundle test: the seven letter-series items, with each model's ability estimated from the 16 verbal items. Totals are summed over models and items (56 model-item pairs); mean: each model's proportion correct, stop-loss test; modal: exact test. Calibrated $p$: $(1+k)/(n+1)$, where $k$ of $n$ null data sets give a bundle $p$-value at least as small as the observed one ($n=1{{,}}500$ in the main arm and 375 at temperature~0; 2{{,}}000 and 1{{,}}000 in column {hathead}). Nulls: the three run-dependence nulls of Table~\\ref{{tab:reductions}}, and a two-dimensional null in which each model's bundle ability differs from its verbal ability as much as two human abilities correlated $\\rho$ (no regression to the mean credited).{hatnote} The instrument's own subscale, all nine LN items, is shown for comparison.}}
    \\label{{tab:bundle}}\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{2pt}}
    \\begin{{tabular}}{{lrrccccc}}\\toprule
     & & & \\multicolumn{{3}}{{c}}{{calibrated $p$}} & \\multicolumn{{2}}{{c}}{{2-D, $\\rho$}} \\\\\\cmidrule(lr){{4-6}}\\cmidrule(lr){{7-8}}
    bundle & obs. & exp. & fixed & fresh & matched & {hathead} & 0.5 \\\\\\midrule
    letter, modal & {M['ckLetModalObs']} & {M['ckLetModalExp']} & {fp(cal['A0']['modal:letter_given_VR']['calibrated_p'],'0.001')} & {fp(cal['B']['modal:letter_given_VR']['calibrated_p'],'0.001')} & {fp(cal['C']['modal:letter_given_VR']['calibrated_p'],'0.001')} & {hat['modal']} & {fp(cal['C_rho0.5']['modal:letter_given_VR']['calibrated_p'],'0.001')} \\\\
    letter, mean & {M['ckLetMeanObs']} & {M['ckLetMeanExp']} & {M['ckLetCalAzero']} & {M['ckLetCalB']} & {M['ckLetCalC']} & {hat['mean']} & {M['ckLetCalCRhoFive']} \\\\
    \\quad temp.\\ 0 & {M['ckTzLetMeanObs']} & {M['ckTzLetMeanExp']} & {M['ckTzLetCalAzero']} & {M['ckTzLetCalB']} & {M['ckTzLetCalC']} & {hat['tz']} & {M['ckTzLetCalCRhoFive']} \\\\
    LN (all 9), mean & {M['ckLNMeanObs']} & {M['ckLNMeanExp']} & {M['ckLNCalAzero']} & {M['ckLNCalB']} & {M['ckLNCalC']} & {hat['ln']} & {M['ckLNCalCRhoFive']} \\\\
    \\bottomrule\\end{{tabular}}\\end{{table}}
    """
    open(os.path.join(T, "tab_bundle.tex"), "w").write(bt)


# ---- Table: run-level nulls (extends tab_runlevel)
def row(lab, v):
    v = np.asarray(v)
    return f"{lab} & {f3(v.mean())} & [{f2(np.quantile(v,.05))}, {f2(np.quantile(v,.95))}] & {f2((v >= 0.84-1e-9).mean())} \\\\"
rl = "\n".join([
    "\\begin{table}[t]",
    f"\\caption{{What the run-level pipeline reports when invariance holds. Human controls: 600 distinct people, 200 replicates. Nulls: eight synthetic humans at the models' abilities, each re-run 75 times with that model's missing cells; {M['ckNullReps']} data sets each. The rows differ in the dependence among a model's runs: none (fresh independent draws); shared item propensities matched to each model's run-to-run variance, centred exactly on the human curve; or one answer copied to every run and flipped at the model's flip rate, the null of Section~\\ref{{sec:runlevel}}, whose flips also pull each item's run proportion towards one half. Observed: 0.84.}}",
    "\\label{tab:runlevel2}\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
    "\\begin{tabular}{lccc}\\toprule",
    "data & mean rate & 90\\% range & $P(\\ge0.84)$ \\\\\\midrule",
    row("random split, people", hc[hc.contrast == "placebo"].rate),
    row("men vs.\\ women, people", hc[hc.contrast == "gender"].rate),
    row("re-runs: independent", n4[n4.kind == "B"].rate),
    row("re-runs: shared propensities", n4[n4.kind == "C"].rate),
    row("re-runs: one answer + flips", a13.rate),
    "\\bottomrule\\end{tabular}\\end{table}", ""])
open(os.path.join(T, "tab_runlevel_v2.tex"), "w").write(rl)

# ---- Table: power
lines = ["\\begin{table}[t]",
         "\\caption{Power of the model-level tests. Items: one item at a time made easier or harder by $d$ logits for all eight simulated examinees at the models' abilities (200 data sets per item, shift and direction; modal-answer test, BH at 0.05). Bundle: all seven letter-series items made harder by $d$ for every model, run-mean bundle test at 0.05, runs dependent as in the matched null (300 data sets).}",
         "\\label{tab:power2}\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{rcccc}\\toprule",
         " & \\multicolumn{2}{c}{item easier for models} & \\multicolumn{2}{c}{item harder for models} \\\\\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "$d$ & median power & items $\\ge0.8$ & median power & items $\\ge0.8$ \\\\\\midrule"]
for d, w in ((1.0, "One"), (2.0, "Two"), (3.0, "Three"), (4.0, "Four")):
    lines.append(f"{d:.0f} & {M['ckPowEasier'+w]} & {M['ckPowNEasier'+w]} & {M['ckPowHarder'+w]} & {M['ckPowNHarder'+w]} \\\\")
lines += ["\\midrule",
          f"\\multicolumn{{5}}{{p{{0.86\\columnwidth}}}}{{Letter-series bundle made harder by $d=0.5$, $0.75$, $1$: power {M['ckBunPowHalf']}, {M['ckBunPowSeventyFive']}, {M['ckBunPowOne']}.}} \\\\",
          "\\bottomrule\\end{tabular}\\end{table}", ""]
open(os.path.join(T, "tab_power_v2.tex"), "w").write("\n".join(lines))
print(open(os.path.join(T, "ck_numbers.tex")).read())
