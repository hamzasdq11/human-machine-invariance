"""End-to-end smoke test: synthetic 'machine' responses through the real pipeline.

Verifies that run_invariance.py executes on data shaped exactly like the harness
output, and that it recovers injected non-invariance.  Run before trusting a real
administration; a pipeline that cannot recover known DIF cannot be trusted to
report unknown DIF.
"""
import os, sys, subprocess, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from dif.irt import prob_2pl, fit_2pl_mml

ROOT = os.path.join(os.path.dirname(__file__), "..")
DER = os.path.join(ROOT, "data", "derived")


def build_fake_machine(inject_items=("letter.58",), b_shift=1.2, a_ratio=0.4,
                       n_resp=300, mu=-0.4, sigma=1.2, seed=3, n_human=1525):
    """Simulate a human reference and machine responses from the ICAR-16 2PL calibration in
    results/icar_calibration.csv (the psychTools response file itself is GPL and not distributed)."""
    meta_all = pd.read_csv(os.path.join(DER, "icar_items.csv"))
    cal = pd.read_csv(os.path.join(ROOT, "results", "icar_calibration.csv")).set_index("item")
    a_all = cal.loc[meta_all["item"], "a"].to_numpy(float)
    b_all = cal.loc[meta_all["item"], "b"].to_numpy(float)
    rng_h = np.random.default_rng(seed + 100)
    th_h = rng_h.normal(0, 1, n_human)               # all 16 items, as the pipeline expects
    human = (rng_h.random((n_human, len(a_all))) < prob_2pl(th_h, a_all, b_all)).astype(float)
    keep = meta_all["text_administrable"].values.astype(bool)
    meta = meta_all[keep].reset_index(drop=True)
    a, b = a_all[keep].copy(), b_all[keep].copy()

    names = meta["item"].tolist()
    for it in inject_items:
        j = names.index(it)
        b[j] -= b_shift          # uniform: memorisation-like
    j2 = names.index(names[-1] if names[-1] not in inject_items else names[0])
    a[j2] *= a_ratio             # non-uniform: divergence-like

    rng = np.random.default_rng(seed)
    th = rng.normal(mu, sigma, n_resp)
    X = (rng.random((n_resp, len(names))) < prob_2pl(th, a, b)).astype(float)

    models = [f"fake/model-{k}" for k in range(4)]
    variants = ["v1", "v2", "v3", "v4", "v5"]
    idx, rows = [], []
    k = 0
    for m in models:
        for v in variants:
            for s in range(n_resp // (len(models) * len(variants))):
                idx.append(f"{m}|{v}|{s}")
                rows.append(dict(model=m, prompt_variant=v, seed=s, parsable_rate=0.98))
                k += 1
    X = X[:len(idx)]
    wide = pd.DataFrame(X, index=idx, columns=names)
    meta_df = pd.DataFrame(rows, index=idx)
    return wide, meta_df, names[names.index(inject_items[0])], names[j2], human


def run_check():
    """Returns True when both injected violations are recovered. Writes nothing to results/."""
    wide, meta_df, uniform_item, nonuniform_item, human = build_fake_machine()
    with tempfile.TemporaryDirectory() as td:
        pq = os.path.join(td, "m.parquet"); cs = os.path.join(td, "m.csv"); hn = os.path.join(td, "h.npy")
        wide.to_parquet(pq); meta_df.to_csv(cs); np.save(hn, human)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_invariance.py"),
                            "--machine", pq, "--meta", cs, "--human", hn, "--out", td],
                           capture_output=True, text=True)
        print(r.stdout[-2500:])
        if r.returncode != 0:
            print("STDERR:", r.stderr[-2000:]); return False
        tab = pd.read_csv(os.path.join(td, "invariance_human_vs_machine.csv"))
        summ = json.load(open(os.path.join(td, "invariance_summary.json")))
    flagged = set(tab.loc[tab.flag_total, "item"])
    ui = dict(zip(tab["item"], tab["uniformity_index"]))

    print("\n--- assertions ---")
    ok = True
    for label, item in (("uniform(injected)", uniform_item), ("non-uniform(injected)", nonuniform_item)):
        hit = item in flagged
        print(f"  {label:24s} {item:12s} flagged={hit}  UI={ui.get(item, float('nan')):.3f}")
        ok &= hit
    print(f"  raw gap {summ.get('raw_gap', float('nan')):+.3f} -> "
          f"purified theta gap {summ.get('purified_theta_gap', float('nan')):+.3f}")
    print(f"  invariant items retained: {summ.get('n_invariant_items')}")
    print("\nRESULT:", "PASS" if ok else "FAIL — injected DIF not recovered")
    return ok


def test_end_to_end():          # pytest entry point
    assert run_check()


if __name__ == "__main__":
    sys.exit(0 if run_check() else 1)
