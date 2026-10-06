"""C5: figures for the camera-ready, drawn from the c2-c4 results files.

fig_items.pdf    model-level item calibration: observed vs expected models correct
fig_letter.pdf   per-model letter-series accuracy vs the accuracy its verbal ability predicts
fig_runlevel.pdf run-level DIF rate under three run-dependence nulls, human controls, observed
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
INK, MUTED, GRID = "#0b0b0b", "#6f6e69", "#d9d8d4"
BLUE, ORANGE = "#2a78d6", "#eb6834"
W1 = 3.45                                              # IEEE column width, inches
EMPH_LETTER = os.environ.get("EMPH_LETTER", "1") == "1"   # Tier 1 builds with 0: item types shown neutrally


def fig_items():
    t = pd.read_csv(os.path.join(C.OUT, "c2_item_tests_main.csv"))
    x, y = t["mean_E"].to_numpy(), t["mean_sum"].to_numpy()
    robust = t["item"].isin(["VR.36", "VR.39"]).to_numpy()
    letter = t["item"].isin(np.array(C.ITEMS)[C.LETTER]).to_numpy()
    number = t["item"].isin(np.array(C.ITEMS)[C.NUMBER]).to_numpy()
    fig, ax = plt.subplots(figsize=(W1, 3.25))
    ax.plot([0, 8], [0, 8], color=MUTED, lw=0.8, zorder=1)
    if not EMPH_LETTER:                                # Tier 1: no item-type split (the bundle analysis is Tier 2)
        letter, number = np.zeros_like(letter), np.zeros_like(number)
    other = ~(robust | letter | number)
    h1 = ax.scatter(x[other], y[other], s=16, facecolor="white", edgecolor=MUTED, lw=0.8, zorder=3)
    h2 = ax.scatter(x[number], y[number], s=16, marker="s", facecolor="white", edgecolor=MUTED, lw=0.8, zorder=3)
    if EMPH_LETTER:
        h3 = ax.scatter(x[letter], y[letter], s=26, marker="^", facecolor=ORANGE, edgecolor="white", lw=0.6, zorder=4)
    else:
        h3 = ax.scatter(x[letter], y[letter], s=18, marker="^", facecolor="white", edgecolor=MUTED, lw=0.8, zorder=4)
    h4 = ax.scatter(x[robust], y[robust], s=28, facecolor=BLUE, edgecolor="white", lw=0.6, zorder=5)
    labels = [("VR.36", 0.22, 0.0, "left"), ("VR.39", 0.0, -0.42, "center"), ("VR.26", -0.2, 0.0, "right"),
              ("VR.42", 0.2, -0.12, "left"), ("LN.01", -0.2, 0.12, "right")]
    if EMPH_LETTER:                                    # Tier 2: name the third antonym item (VR.11) too
        labels.append(("VR.11", -0.2, 0.0, "right"))
    for it, dx, dy, ha in labels:
        r = t[t.item == it].iloc[0]
        ax.annotate(it, (r.mean_E, r.mean_sum), xytext=(r.mean_E + dx, r.mean_sum + dy), ha=ha, va="center",
                    fontsize=7, color=INK if it in ("VR.36", "VR.39") else MUTED)
    if EMPH_LETTER:
        ax.text(4.3, 0.45, "letter series: 6 of 7\nitems below the line", fontsize=6.8, color=INK, va="bottom")
        ax.annotate("", xy=(3.88, 1.02), xytext=(4.25, 0.72), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlim(0, 8.2); ax.set_ylim(0, 8.2)
    ax.set_xticks(range(0, 9, 2)); ax.set_yticks(range(0, 9, 2))
    ax.set_aspect("equal")
    ax.set_xlabel("Expected if invariant (models correct, of 8)")
    ax.set_ylabel("Observed (models correct per run, of 8)")
    if EMPH_LETTER:
        fig.legend([h1, h2, h3, h4], ["verbal", "number series", "letter series", "flagged under all three reductions"],
                   loc="upper center", ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.0,
                   bbox_to_anchor=(0.55, 1.0), fontsize=7)
        fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    else:
        fig.legend([h1, h4], ["other items", "flagged under all three reductions"],
                   loc="upper center", ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.0,
                   bbox_to_anchor=(0.55, 1.0), fontsize=7)
        fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "fig_items.pdf")); fig.savefig(os.path.join(FIG, "fig_items.png"))
    plt.close(fig)


def fig_letter():
    pm = pd.DataFrame(json.load(open(os.path.join(C.OUT, "c3_bundles.json")))["main"]["per_model"]).set_index("model")
    fig, ax = plt.subplots(figsize=(W1, 2.55))
    ax.plot([0, 0.55], [0, 0.55], color=MUTED, lw=0.8, zorder=1)
    q = pm.loc[["Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-14B"]]
    ax.plot(q.letter_expected_from_VR, q.letter_mean, color=ORANGE, lw=0.8, alpha=0.55, zorder=2)
    ax.scatter(pm.letter_expected_from_VR, pm.letter_mean, s=24, marker="^", facecolor=ORANGE, edgecolor="white", lw=0.6, zorder=3)
    place = {"Meta-Llama-3.1-8B": ("Llama-3.1-8B", (0.175, 0.215), "right"),
             "OpenHermes-2.5-Mistral-7B": ("OpenHermes-2.5", (0.25, 0.168), "left"),
             "Phi-3.5-mini-instruct": ("Phi-3.5-mini", (0.315, 0.135), "left"),
             "OLMo-2-13B": ("OLMo-2-13B", (0.155, 0.115), "right"),
             "Qwen2.5-3B": ("Qwen2.5-3B", (0.155, 0.055), "right"),
             "OLMo-2-7B": ("OLMo-2-7B", (0.29, 0.035), "left"),
             "Qwen2.5-7B": ("Qwen2.5-7B", (0.36, 0.115), "center"),
             "Qwen2.5-14B": ("Qwen2.5-14B", (0.47, 0.085), "center")}
    for m, (lab, (tx, ty), ha) in place.items():
        r = pm.loc[m]
        ax.annotate(lab, (r.letter_expected_from_VR, r.letter_mean), xytext=(tx, ty), fontsize=6.5, color=INK, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6, shrinkA=1, shrinkB=3))
    p0, p1 = ax.transData.transform((0.2, 0.2)), ax.transData.transform((0.3, 0.3))
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(0.262, 0.290, "invariance", fontsize=6.5, color=MUTED, rotation=ang, ha="center", va="center", rotation_mode="anchor")
    ax.set_xlim(0, 0.55); ax.set_ylim(0, 0.32)
    ax.set_xlabel("Letter-series accuracy expected from verbal ability")
    ax.set_ylabel("Observed letter-series accuracy")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_letter.pdf")); fig.savefig(os.path.join(FIG, "fig_letter.png"))
    plt.close(fig)


def fig_runlevel():
    n4 = pd.read_csv(os.path.join(C.OUT, "c4_runlevel_nulls.csv"))
    hc = pd.read_csv(os.path.join(C.ROOT, "results", "a6_matched_controls25.csv"))
    a13 = pd.read_csv(os.path.join(C.ROOT, "results", "a13_replicated_human_null25.csv"))   # the paper's own null
    rows = [("Random split, 600 people", hc[hc.contrast == "placebo"].rate.to_numpy(), "#9c9b96"),
            ("Men vs women, 600 people", hc[hc.contrast == "gender"].rate.to_numpy(), "#9c9b96"),
            ("Re-runs: independent", n4[n4.kind == "B"].rate.to_numpy(), "#9c9b96"),
            ("Re-runs: shared propensities", n4[n4.kind == "C"].rate.to_numpy(), "#9c9b96"),
            ("Re-runs: one answer + flips", a13.rate.to_numpy(), "#9c9b96")]
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(W1, 1.95))
    for k, (lab, v, col) in enumerate(rows[::-1]):
        yk = k
        ax.scatter(v + rng.uniform(-0.006, 0.006, len(v)), yk + rng.uniform(-0.18, 0.18, len(v)), s=4, color=col, alpha=0.55, lw=0, zorder=2)
        ax.plot([np.median(v)] * 2, [yk - 0.3, yk + 0.3], color=INK, lw=1.2, zorder=3)
    ax.axvline(0.84, color=BLUE, lw=1.2, zorder=4)
    ax.text(0.835, len(rows) - 0.45, "8 models,\n600 runs: 0.84", color=BLUE, fontsize=6.8, ha="right", va="top")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Share of 25 items flagged, run-level pipeline")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_runlevel.pdf")); fig.savefig(os.path.join(FIG, "fig_runlevel.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_items(); fig_letter(); fig_runlevel()
    print("written:", sorted(os.listdir(FIG)))
