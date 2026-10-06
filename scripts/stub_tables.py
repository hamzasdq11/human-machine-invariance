"""Emit placeholder fragments so the paper always compiles, even mid-pipeline."""
import os
T = os.path.join(os.path.dirname(__file__), "..", "paper", "tables")
os.makedirs(T, exist_ok=True)
STUBS = {
 "tab_separation.tex": ("separation envelope", "tab:separation"),
 "tab_focaln.tex": ("focal-size study", "tab:focaln"),
 "tab_placebo.tex": ("random-split placebo", "tab:placebo"),
 "tab_subgroup.tex": ("human-subgroup DIF", "tab:subgroup"),
 "tab_icar.tex": ("ICAR-16 calibration", "tab:icar"),
 "tab_anchor.tex": ("anchor-parameter uncertainty", "tab:anchor"),
 "tab_subgroup_matched.tex": ("matched-N subgroup DIF", "tab:subgroupmatched"),
}
DEFAULTS = dict(envLo="-2.0", envHi="+1.0", sepReps="0", typeOneSmall="0.000",
    powerAtHundred="0.000", nullTypeOne="0.000", nullLo="0.000", nullHi="0.000", nullReps="0",
    uiUniform="0.000", uiNonuniform="0.000", aucKind="0.000", placeboMean="0.000",
    placeboLo="0.000", placeboHi="0.000", placeboN="0", subgroupMin="0.000",
    subgroupMax="0.000", nTextItems="8", icarN="1525",
    safeRatio="0.20", inflatedTypeOne="0.000", oracleTypeOne="0.000", maxNfoc="3200",
    subMatchedMin="0.000", subMatchedMax="0.000", subMatchedEtsMin="0.000",
    subMatchedEtsMax="0.000", subMatchedN="300", placeboAtMatched="0.000",
    placeboMatchedLo="0.000", placeboMatchedHi="0.000", placeboEvenSplit="0.000",
    uiNone="0.000")
for f, (desc, lab) in STUBS.items():
    p = os.path.join(T, f)
    if not os.path.exists(p):
        open(p, "w").write(
            "\\begin{table}[t]\\caption{%s --- \\PENDING{pending}}\\label{%s}\\centering "
            "\\begin{tabular}{c}\\toprule pending \\\\ \\bottomrule\\end{tabular}"
            "\\end{table}\n" % (desc, lab))
mp = os.path.join(T, "macros.tex")
have = {}
if os.path.exists(mp):
    for line in open(mp):
        if line.startswith("\\newcommand{\\"):
            have[line.split("{\\", 1)[1].split("}", 1)[0]] = line
with open(mp, "w") as f:
    for k, v in DEFAULTS.items():
        f.write(have.get(k, "\\newcommand{\\%s}{\\PENDING{?}}\n" % k) if k not in have
                else have[k])
    for k, line in have.items():
        if k not in DEFAULTS: f.write(line)
print("stubs ensured")
