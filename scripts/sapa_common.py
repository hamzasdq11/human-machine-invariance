"""Shared loaders for the 25-item SAPA analysis (rebuilt 3 Oct 2026).

The August scripts that produced the accepted paper's 25-item results did not
survive; the estimation library in dif/ did. These loaders rebuild the data
layer around that library. Every published number is a regression test for them
(see scripts/reproduce_gate1.py).
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

ITEMS = ["VR.04", "VR.09", "VR.11", "VR.13", "VR.14", "VR.16", "VR.17", "VR.18",
         "VR.19", "VR.23", "VR.26", "VR.31", "VR.32", "VR.36", "VR.39", "VR.42",
         "LN.01", "LN.03", "LN.05", "LN.06", "LN.07", "LN.33", "LN.34", "LN.35",
         "LN.58"]

# harness item_id -> ICAR id
HARNESS_TO_ICAR = {
    "reason.4": "VR.04", "vr_09": "VR.09", "vr_11": "VR.11", "vr_13": "VR.13",
    "vr_14": "VR.14", "reason.16": "VR.16", "reason.17": "VR.17", "vr_18": "VR.18",
    "reason.19": "VR.19", "vr_23": "VR.23", "vr_26": "VR.26", "vr_31": "VR.31",
    "vr_32": "VR.32", "vr_36": "VR.36", "vr_39": "VR.39", "vr_42": "VR.42",
    "ln_01": "LN.01", "ln_03": "LN.03", "ln_05": "LN.05", "ln_06": "LN.06",
    "letter.7": "LN.07", "letter.33": "LN.33", "letter.34": "LN.34",
    "ln_35": "LN.35", "letter.58": "LN.58",
}

SAPA_CSV = os.path.join(DATA, "sapaICARData18aug2010thru20may2013.csv")


def load_sapa():
    """SAPA ICAR release restricted to respondents with >=1 of the 25 items.

    Returns (R, demo): R is (95166, 25) float with NaN for not-administered.
    """
    d = pd.read_csv(SAPA_CSV, index_col=0)
    X = d[ITEMS]
    keep = X.notna().any(axis=1).values
    return X.values[keep].astype(float), d.loc[keep, ["gender", "age"]].reset_index(drop=True)


def load_reference_params(path=None):
    path = path or os.path.join(RES, "sapa_calibration.csv")
    c = pd.read_csv(path).set_index("item").loc[ITEMS]
    return c["a"].values, c["b"].values


def load_machine(path, min_parse=0.80):
    """Machine responses -> respondent x item matrix.

    One respondent = one (model, prompt_variant, seed) triple. Unparsable
    responses are missing, never scored incorrect.
    Returns (R, meta, long_df).
    """
    df = pd.read_json(path, lines=True)
    df["icar"] = df["item_id"].map(HARNESS_TO_ICAR)
    assert df["icar"].notna().all(), "unmapped item ids"
    df["y"] = np.where(df["parsable"].astype(bool), df["correct"].astype(float), np.nan)
    pr = df.groupby("model")["parsable"].mean()
    excluded = list(pr[pr < min_parse].index)
    df = df[~df["model"].isin(excluded)]
    wide = df.pivot_table(index=["model", "prompt_variant", "seed"], columns="icar",
                          values="y", aggfunc="first", dropna=False)
    wide = wide.reindex(columns=ITEMS)
    meta = wide.index.to_frame(index=False)
    return wide.values.astype(float), meta, df, excluded
