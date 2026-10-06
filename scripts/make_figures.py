"""Figures for the paper. Print-targeted, colourblind-safe, generated from results."""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
F = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(F, exist_ok=True)

# Validated categorical palette (six checks pass, light surface):
#   node scripts/validate_palette.js "#1F5FA9,#C2571A,#8E4B9E" --mode light
BLUE, ORANGE, PURPLE = "#1F5FA9", "#C2571A", "#8E4B9E"
INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#D8D8D6"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "figure.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def figure1():
    """Two panels: operating envelope, and separability of the two DIF kinds."""
    pa = os.path.join(R, "sim_A_separation.csv")
    if not os.path.exists(pa):
        print("  [skip fig1] no separation results yet"); return
    d = pd.read_csv(pa)

    # IEEE two-column: 7.16in full width; use ~3.4in single column x 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))

    g = d.groupby("mu_true").agg(power=("power", "mean"), type1=("type1", "mean"),
                                 n=("power", "size")).reset_index()
    # bootstrap CIs
    def bci(col, mu):
        x = d.loc[d.mu_true == mu, col].values
        rng = np.random.default_rng(0)
        bs = rng.choice(x, size=(2000, len(x))).mean(1)
        return np.quantile(bs, [.025, .975])
    for col, colr, lab, mk in ((("power"), BLUE, "Power (injected DIF detected)", "o"),
                               (("type1"), ORANGE, "Type-I (DIF-free items flagged)", "s")):
        lo = np.array([bci(col, m)[0] for m in g.mu_true])
        hi = np.array([bci(col, m)[1] for m in g.mu_true])
        ax1.fill_between(g.mu_true, lo, hi, color=colr, alpha=0.15, linewidth=0)
        ax1.plot(g.mu_true, g[col], color=colr, linewidth=2.0, marker=mk,
                 markersize=4.5, markeredgecolor="white", markeredgewidth=0.7,
                 label=lab, clip_on=False)
    ax1.axhline(0.05, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax1.text(g.mu_true.min(), 0.075, r"nominal $\alpha=0.05$", fontsize=6.5, color=MUTED)
    ax1.set_xlabel(r"focal-group ability offset $\mu$  (reference $=0$)")
    ax1.set_ylabel("rate")
    ax1.set_ylim(-0.02, 1.05); ax1.set_title("(a) Operating envelope", loc="left")
    ax1.legend(frameon=False, loc="center left", handlelength=1.6)
    _clean(ax1)

    # panel b: uniformity index by injected kind
    cats = [("uniform\n(difficulty shift)", "ui_uniform", BLUE),
            ("non-uniform\n(discrimination)", "ui_nonuniform", ORANGE),
            ("no DIF", "ui_none", PURPLE)]
    data = [d[c].dropna().values for _, c, _ in cats]
    bp = ax2.boxplot(data, vert=False, widths=0.55, patch_artist=True,
                     showfliers=False, medianprops=dict(color="white", linewidth=1.6))
    for patch, (_, _, colr) in zip(bp["boxes"], cats):
        patch.set_facecolor(colr); patch.set_edgecolor("white"); patch.set_linewidth(1.2)
    for el in ("whiskers", "caps"):
        for art in bp[el]:
            art.set_color(MUTED); art.set_linewidth(0.8)
    ax2.set_yticklabels([c[0] for c in cats])
    ax2.set_xlabel(r"uniformity index  $\mathrm{UI}=\Lambda^{\mathrm{unif}}/\Lambda^{\mathrm{tot}}$")
    ax2.set_xlim(-0.02, 1.02)
    auc = d["auc_kind"].mean()
    ax2.set_title(f"(b) Separability of the two kinds  (AUC $=$ {auc:.2f})", loc="left")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.grid(True, axis="x", color=GRID, linewidth=0.6); ax2.set_axisbelow(True)

    out = os.path.join(F, "fig_validation.pdf")
    fig.tight_layout(w_pad=2.0); fig.savefig(out); plt.close(fig)
    print("wrote", out)


def figure2():
    """Focal sample size: the regime-independence claim."""
    pb = os.path.join(R, "sim_B_focal_n.csv")
    if not os.path.exists(pb):
        print("  [skip fig2] no focal-N results yet"); return
    d = pd.read_csv(pb)
    g = d.groupby("n_foc").agg(power=("power", "mean"), type1=("type1", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.plot(g.n_foc, g.power, color=BLUE, linewidth=2.0, marker="o", markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.7, label="power", clip_on=False)
    ax.plot(g.n_foc, g.type1, color=ORANGE, linewidth=2.0, marker="s", markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.7, label="Type-I", clip_on=False)
    ax.axhline(0.05, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.axvline(100, color=MUTED, linewidth=0.8, linestyle=(0, (1, 2)))
    ax.text(105, 0.55, "direct calibration\nreported unreliable\nbelow this point",
            fontsize=6, color=MUTED, va="center")
    ax.set_xscale("log"); ax.set_xticks(g.n_foc)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("focal-group respondents"); ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, loc="center right", handlelength=1.6)
    _clean(ax)
    out = os.path.join(F, "fig_focaln.pdf")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    figure1(); figure2()
