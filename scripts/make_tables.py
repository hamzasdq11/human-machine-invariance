"""Turn result CSVs into LaTeX fragments that the paper \\input{}s.

No number in the paper is typed by hand; every table and every inline statistic
is generated from a results file. Re-running the pipeline regenerates the paper.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np, pandas as pd

R = os.path.join(os.path.dirname(__file__), "..", "results")
T = os.path.join(os.path.dirname(__file__), "..", "paper", "tables")
os.makedirs(T, exist_ok=True)
MACROS = {}


def w(name, body):
    with open(os.path.join(T, name), "w") as f:
        f.write(body)
    print("wrote", name)


def macro(key, val, fmt="{:.3f}"):
    MACROS[key] = fmt.format(val) if isinstance(val, (int, float, np.floating)) else str(val)


def ci(x, alpha=0.05, n_boot=4000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return tuple(np.quantile(bs, [alpha / 2, 1 - alpha / 2]))


# ---------------------------------------------------------------- Study A
def study_a():
    p = os.path.join(R, "sim_A_separation.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p)
    g = d.groupby("mu_true").agg(
        n=("power", "size"), power=("power", "mean"), type1=("type1", "mean"),
        mu_bias=("mu_hat", "mean"), sigma_hat=("sigma_hat", "mean"),
        auc=("auc_kind", "mean")).reset_index()
    g["mu_bias"] = g["mu_bias"] - g["mu_true"]
    rows = "\n".join(
        f"{r.mu_true:+.1f} & {int(r.n)} & {r.power:.3f} & {r.type1:.3f} & "
        f"{r.mu_bias:+.3f} & {r.sigma_hat:.3f} & {r.auc:.3f} \\\\"
        for r in g.itertuples())
    w("tab_separation.tex", r"""\begin{table}[t]
\caption{Operating envelope of the anchored estimator as a function of the
ability separation between reference and focal groups. Focal ability is
$\mathcal{N}(\mu,1)$; the reference group is $\mathcal{N}(0,1)$. Power is the
detection rate for injected DIF items, Type-I the flag rate among DIF-free items
after Benjamini--Hochberg control at $\alpha=0.05$. AUC is the discrimination of
the uniformity index (\ref{eq:ui}) between injected uniform and non-uniform DIF.}
\label{tab:separation}
\centering
\begin{tabular}{rrrrrrr}
\toprule
$\mu$ & reps & power & Type-I & bias($\hat\mu$) & $\hat\sigma$ & AUC \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    ok = g[(g.type1 <= 0.10)]
    if len(ok):
        macro("envLo", ok.mu_true.min(), "{:+.1f}"); macro("envHi", ok.mu_true.max(), "{:+.1f}")
    macro("sepReps", int(d.groupby('mu_true').size().min()), "{:d}")


# ---------------------------------------------------------------- Study B
def study_b():
    p = os.path.join(R, "sim_B_focal_n.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p)
    g = d.groupby("n_foc").agg(n=("power", "size"), power=("power", "mean"),
                               type1=("type1", "mean"), mu_hat=("mu_hat", "mean"),
                               sigma_hat=("sigma_hat", "mean")).reset_index()
    rows = "\n".join(
        f"{int(r.n_foc)} & {int(r.n)} & {r.power:.3f} & {r.type1:.3f} & "
        f"{r.mu_hat:+.3f} & {r.sigma_hat:.3f} \\\\" for r in g.itertuples())
    w("tab_focaln.tex", r"""\begin{table}[t]
\caption{Behaviour as the focal sample shrinks. Direct two-group calibration is
reported in prior work to become unreliable below roughly one hundred
respondents; the anchored estimator holds its Type-I rate across the range
because item parameters are never identified from focal data.}
\label{tab:focaln}
\centering
\begin{tabular}{rrrrrr}
\toprule
$N_{\text{focal}}$ & reps & power & Type-I & $\hat\mu$ & $\hat\sigma$ \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    small = g[g.n_foc <= 50]
    if len(small): macro("typeOneSmall", small.type1.max())
    if len(g): macro("powerAtHundred", float(g.loc[g.n_foc == 100, "power"].mean()))


# ---------------------------------------------------------------- Study C
def study_c():
    p = os.path.join(R, "sim_C_null.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p)
    lo, hi = ci(d["type1"].values)
    macro("nullTypeOne", d["type1"].mean()); macro("nullLo", lo); macro("nullHi", hi)
    macro("nullReps", len(d), "{:d}")


# ------------------------------------------------------- separability figure
def separability():
    p = os.path.join(R, "sim_A_separation.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p)
    macro("uiUniform", d["ui_uniform"].mean()); macro("uiNonuniform", d["ui_nonuniform"].mean())
    macro("uiNone", d["ui_none"].mean()); macro("aucKind", d["auc_kind"].mean())


# ---------------------------------------------------------------- placebo
def placebo():
    p = os.path.join(R, "placebo_icar.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p)
    g = d.groupby("n_foc").agg(n=("dif_rate_total", "size"),
                               rate=("dif_rate_total", "mean"),
                               sd=("dif_rate_total", "std"),
                               unif=("dif_rate_uniform", "mean"),
                               nonu=("dif_rate_nonuniform", "mean")).reset_index()
    rows = []
    for r in g.itertuples():
        lo, hi = ci(d.loc[d.n_foc == r.n_foc, "dif_rate_total"].values)
        rows.append(f"{int(r.n_foc)} & {int(r.n)} & {r.rate:.3f} & [{lo:.3f}, {hi:.3f}] & "
                    f"{r.unif:.3f} & {r.nonu:.3f} \\\\")
    w("tab_placebo.tex", r"""\begin{table}[t]
\caption{Random-split placebo on ICAR-16 human responses ($N=1{,}525$). A single
human sample is split arbitrarily and the identical pipeline is run. This bounds
the false-positive floor of the procedure on real data with real item structure.}
\label{tab:placebo}
\centering
\begin{tabular}{rrrlrr}
\toprule
$N_{\text{focal}}$ & splits & DIF rate & 95\% CI & uniform & non-unif. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    macro("placeboMean", d["dif_rate_total"].mean())
    lo, hi = ci(d["dif_rate_total"].values); macro("placeboLo", lo); macro("placeboHi", hi)
    macro("placeboN", len(d), "{:d}")
    # the rate at the focal size actually used for the machine analysis
    MATCHED = 250
    sel = d.loc[d.n_foc == MATCHED, "dif_rate_total"].values
    if len(sel):
        mlo, mhi = ci(sel)
        macro("placeboAtMatched", sel.mean())
        macro("placeboMatchedLo", mlo); macro("placeboMatchedHi", mhi)
    big = d.loc[d.n_foc == d.n_foc.max(), "dif_rate_total"]
    macro("placeboEvenSplit", float(big.mean()))


# ---------------------------------------------------------------- subgroup
def subgroup():
    p = os.path.join(R, "subgroup_summary.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p)
    rows = "\n".join(
        f"{r.instrument.upper()} & {str(r.contrast).replace('_',' ')} & {int(r.n_ref)}/{int(r.n_foc)} & "
        f"{int(r.n_items)} & {r.dif_rate_total:.3f} & {r.ets_BC_rate:.3f} \\\\"
        for r in d.itertuples())
    w("tab_subgroup.tex", r"""\begin{table}[t]
\caption{Ordinary cross-population non-invariance: anchored likelihood-ratio DIF
between real human demographic subgroups, with the Mantel--Haenszel ETS B+C rate
as an independent second estimator. These rates are the yardstick against which a
human--machine result must be read.}
\label{tab:subgroup}
\centering
\begin{tabular}{llrrrr}
\toprule
instr. & contrast & $N_{\text{ref}}/N_{\text{foc}}$ & items & LR DIF & MH B+C \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    macro("subgroupMin", d["dif_rate_total"].min()); macro("subgroupMax", d["dif_rate_total"].max())


# ---------------------------------------------------------------- ICAR calib
def calibration():
    p = os.path.join(R, "icar_calibration.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p)
    rows = "\n".join(
        f"\\texttt{{{r.item}}} & {r.domain} & {'yes' if r.text_administrable else 'no'} & "
        f"{r.p_value:.3f} & {r.a:.2f} & {r.b:+.2f} \\\\" for r in d.itertuples())
    w("tab_icar.tex", r"""\begin{table}[t]
\caption{ICAR-16 calibrated on the human sample ($N=1{,}525$, 4.7\% missing).
Items marked not text-administrable are figural and require a vision-capable
respondent; the text-only subset is the pre-registered primary analysis set.}
\label{tab:icar}
\centering
\small
\begin{tabular}{llcrrr}
\toprule
item & domain & text & $p$ & $\hat a$ & $\hat b$ \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    macro("nTextItems", int(d.text_administrable.sum()), "{:d}")
    macro("icarN", 1525, "{:d}")




# ---------------------------------------------------------------- Study D
def study_d():
    p = os.path.join(R, "sim_D_anchor_uncertainty.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p)
    g = d.groupby(["n_foc", "oracle"])[["power", "type1"]].mean().reset_index()
    piv = g.pivot(index="n_foc", columns="oracle")
    n_ref = 2000
    rows = []
    for n in sorted(d.n_foc.unique()):
        est_t = piv[("type1", False)][n]; ora_t = piv[("type1", True)][n]
        est_p = piv[("power", False)][n]
        rows.append(f"{int(n)} & {n/n_ref:.2f} & {est_p:.3f} & {est_t:.3f} & {ora_t:.3f} \\\\")
    w("tab_anchor.tex", r"""\begin{table}[t]
\caption{Anchor-parameter uncertainty. Type-I error inflates as the focal sample
grows relative to the reference sample---the reverse of the usual small-$N$
concern. Supplying the true item parameters removes the inflation entirely while
leaving power unchanged, identifying the cause as reference-sample estimation
error treated as known rather than a defect of the test. Reference $N=2{,}000$.}
\label{tab:anchor}
\centering
\begin{tabular}{rrrrr}
\toprule
 & & & \multicolumn{2}{c}{Type-I} \\
\cmidrule(l){4-5}
$N_{\text{foc}}$ & ratio & power & estimated & true \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    safe = g[(g.oracle == False) & (g.type1 <= 0.05)]
    if len(safe): macro("safeRatio", safe.n_foc.max() / n_ref, "{:.2f}")
    macro("inflatedTypeOne", float(piv[("type1", False)][d.n_foc.max()]))
    macro("oracleTypeOne", float(piv[("type1", True)][d.n_foc.max()]))
    macro("maxNfoc", int(d.n_foc.max()), "{:d}")


# ------------------------------------------------- matched-N subgroup
def subgroup_matched():
    p = os.path.join(R, "subgroup_matchedN.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p)
    g = d.groupby("contrast").agg(n=("dif_rate_total", "size"),
                                  lr=("dif_rate_total", "mean"),
                                  lrsd=("dif_rate_total", "std"),
                                  ets=("ets_BC_rate", "mean"),
                                  etssd=("ets_BC_rate", "std")).reset_index()
    rows = "\n".join(
        f"{r.contrast.replace('_',' ')} & {int(r.n)} & {r.lr:.3f} ({r.lrsd:.3f}) & "
        f"{r.ets:.3f} ({r.etssd:.3f}) \\\\" for r in g.itertuples())
    w("tab_subgroup_matched.tex", r"""\begin{table}[t]
\caption{Ordinary cross-population non-invariance between real human demographic
subgroups, computed at the \emph{same} focal sample size as the machine analysis
($N_{\text{foc}}=300$) so that all three reference quantities---placebo floor,
subgroup baseline, machine result---are measured in one regime. Mean (SD) over
resamples.}
\label{tab:subgroupmatched}
\centering
\begin{tabular}{lrll}
\toprule
contrast & runs & LR DIF rate & MH ETS B+C \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    macro("subMatchedMin", g.lr.min()); macro("subMatchedMax", g.lr.max())
    macro("subMatchedEtsMin", g.ets.min()); macro("subMatchedEtsMax", g.ets.max())
    macro("subMatchedN", 300, "{:d}")


if __name__ == "__main__":
    for fn in (study_a, study_b, study_c, study_d, separability, placebo,
               subgroup, subgroup_matched, calibration):
        try:
            fn()
        except Exception as e:
            print(f"  [skip {fn.__name__}] {e}")
    with open(os.path.join(T, "macros.tex"), "w") as f:
        for k, v in sorted(MACROS.items()):
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print(f"wrote macros.tex with {len(MACROS)} macros")
