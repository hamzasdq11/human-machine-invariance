"""Load and validate the human item-level response matrices.

Sources (all public, fetched from the read-only CRAN mirror on GitHub):
  ability / iqitems  - ICAR-16, N=1525, 16 cognitive-ability items (SAPA project)
  bfi                - Big Five Inventory, N=2800, 25 items + gender/education/age
  spi                - SAPA Personality Inventory, N=4000, 135 items + demographics

ICAR-16 is the primary instrument.  BFI and SPI supply the human-subgroup DIF
calibration baseline (what does ordinary cross-population non-invariance look
like on a psychological instrument?), which the machine-focal analysis is read
against.
"""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd, pyreadr

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DER = os.path.join(os.path.dirname(__file__), "..", "data", "derived")
os.makedirs(DER, exist_ok=True)

ICAR_DOMAIN = {"reason": "verbal reasoning", "letter": "letter/number series",
               "matrix": "matrix reasoning", "rotate": "three-dimensional rotation"}
# text-administrable to a language model without vision:
TEXT_DOMAINS = ("reason", "letter")
# Scoring keys are not distributed with this repository and are not needed here (ability.rda is
# already scored). scripts/verify_items.py reads the published ICAR-16 key from ICAR16_KEYS.


def _read(name, obj=None):
    r = pyreadr.read_r(os.path.join(RAW, name))
    return r[obj] if obj else list(r.values())[0]


def load_icar():
    ab = _read("ability.rda", "ability")
    iq = _read("iqitems.rda", "iqitems")
    ab.columns = [str(c) for c in ab.columns]
    meta = pd.DataFrame({
        "item": ab.columns,
        "domain": [ICAR_DOMAIN[c.split(".")[0]] for c in ab.columns],
        "text_administrable": [c.split(".")[0] in TEXT_DOMAINS for c in ab.columns],
        "p_value": ab.mean(skipna=True).values,
        "n_obs": ab.notna().sum().values,
    })
    return ab.to_numpy(dtype=float), meta, iq


def dichotomise(df, item_cols):
    """Median-split polytomous Likert items into 0/1 (endorsement above median).

    Required because the DIF machinery here is dichotomous.  Applied to the
    *pooled* sample so the split point cannot itself create group differences.
    """
    X = df[item_cols].to_numpy(dtype=float)
    med = np.nanmedian(X, axis=0, keepdims=True)
    B = np.where(np.isnan(X), np.nan, (X > med).astype(float))
    keep = (np.nanmean(B, axis=0) > 0.05) & (np.nanmean(B, axis=0) < 0.95)
    return B[:, keep], [c for c, k in zip(item_cols, keep) if k]


def load_bfi():
    d = _read("bfi.rda", "bfi")
    items = [c for c in d.columns if c not in ("gender", "education", "age")]
    B, kept = dichotomise(d, items)
    return B, kept, d[["gender", "education", "age"]].reset_index(drop=True)


def load_spi():
    d = _read("spi.rda", "spi")
    items = [c for c in d.columns if str(c).startswith("q_")]
    B, kept = dichotomise(d, items)
    demo = d[["age", "sex", "education", "health"]].reset_index(drop=True)
    return B, kept, demo


if __name__ == "__main__":
    R, meta, iq = load_icar()
    meta.to_csv(os.path.join(DER, "icar_items.csv"), index=False)
    np.save(os.path.join(DER, "icar_responses.npy"), R)
    print("ICAR-16:", R.shape, "missing %.2f%%" % (100 * np.isnan(R).mean()))
    print(meta.to_string(index=False))

    B, kept, demo = load_bfi()
    np.save(os.path.join(DER, "bfi_responses.npy"), B)
    demo.to_csv(os.path.join(DER, "bfi_demo.csv"), index=False)
    print("\nBFI:", B.shape, "items kept:", len(kept),
          "| gender counts:", demo.gender.value_counts().to_dict())

    S, skept, sdemo = load_spi()
    np.save(os.path.join(DER, "spi_responses.npy"), S)
    sdemo.to_csv(os.path.join(DER, "spi_demo.csv"), index=False)
    print("SPI:", S.shape, "items kept:", len(skept),
          "| sex counts:", sdemo.sex.value_counts().to_dict())
