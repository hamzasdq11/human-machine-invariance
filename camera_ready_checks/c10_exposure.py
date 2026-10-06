"""C10: are items that appear in open pretraining corpora easier for the models? (Tier 2, plan item B3c)

NOT RUN for the camera-ready: the infini-gram API (https://api.infini-gram.io/) is not reachable from the
analysis environment. Run it on any machine with internet access, from the repository root:

    python3 camera_ready_checks/c9_item_properties.py      # if results/c9_item_properties.json is missing
    python3 camera_ready_checks/c10_exposure.py              # ~250 API calls, a few minutes
    python3 camera_ready_checks/c10_exposure.py --dry-run    # print the queries only, no network

Exposure of an item = the number of exact occurrences, in each corpus index, of its distinctive text:
  * the stem as administered (all items);
  * for series items, the series itself, with and without commas (for example "B, E, H, K" and "B E H K");
  * for the two odd-one-out items (VR.09, VR.23), whose stems are generic, the co-occurrence of all
    six substantive options (a CNF AND query: adjacent words within 100 tokens).
Counts are tokenised n-gram matches (Llama-2 tokenizer), as infini-gram defines them.

Indexes: the OLMo-2-13B-Instruct training data (pre- and post-training; the pool's two OLMo-2 models were
trained on the same data mixes) and four open web corpora. Primary exposure is ICAR-specific (the stem; for
the odd-one-out items, their options together); secondary exposure also counts the bare series, which can
occur outside ICAR (square numbers, Lucas numbers, letter runs). The test: Spearman correlation, over the 25 items, between log(1 + exposure)
and the implied logit shift of c9 (positive = easier for the models than the human curve predicts), with a
permutation p-value; and the same for the two OLMo-2 models alone against their own training data.
Under memorisation, exposed items should be easier for the models: a positive correlation.

Outputs: results/c10_exposure_counts.csv, results/c10_exposure.json
"""
import os, sys, json, time, argparse, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import common as C

API = "https://api.infini-gram.io/"
INDEXES = ["v4_olmo-2-1124-13b-instruct_llama",   # OLMo-2-13B-Instruct, pre- and post-training data
           "v4_dclm-baseline_llama",              # DCLM-baseline, 4.3T tokens of web text
           "v4_dolma-v1_7_llama",                 # Dolma v1.7, 2.6T
           "v4_rpj_llama_s4",                     # RedPajama, 1.4T
           "v4_piletrain_llama"]                  # the Pile (train), 0.4T
OLMO_INDEX = INDEXES[0]
# Item text is not distributed with this repository (PsychArchives Scientific Use Licence,
# doi:10.23668/psycharchives.22167). Point ICAR_ITEMS at your own transcription of the 25 items,
# in the schema described in data/README.md.
_ITEMS = os.environ.get("ICAR_ITEMS")
if not _ITEMS or not os.path.exists(_ITEMS):
    sys.exit("c10 needs the ICAR item text: set ICAR_ITEMS to a JSON file of the 25 items (see data/README.md).")
BANK = {it["icar_id"]: it for it in json.load(open(_ITEMS))}
ODD_ONE_OUT = ("VR.09", "VR.23")


def queries(item):
    it = BANK[item]
    q = [("stem", it["stem"])]
    if item.startswith("LN"):
        series = it["stem"].split("?", 1)[1].strip().rstrip(".").rstrip(",").strip()   # e.g. "B, E, H, K, ..."
        series = series.replace("...", "").strip().rstrip(",").strip()
        q += [("series", series), ("series_nocomma", series.replace(",", ""))]
    if item in ODD_ONE_OUT:
        subst = it["options"][:len(it["options"]) - it["n_fixed_tail"]]            # the six substantive options
        q += [("options_and", " AND ".join(subst))]
    return q


def ask(index, query, tries=6):
    body = json.dumps({"index": index, "query_type": "count", "query": query}).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read().decode())
            if "error" in out:
                raise RuntimeError(out["error"])
            return dict(count=int(out["count"]), approx=bool(out.get("approx", False)), ntok=len(out.get("token_ids", [])))
        except (urllib.error.URLError, RuntimeError, TimeoutError, ValueError) as e:
            if k == tries - 1:
                print(f"  failed after {tries} tries: {index} | {query[:60]} | {e}", flush=True)
                return dict(count=np.nan, approx=None, ntok=np.nan)
            time.sleep(2 ** k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = []
    for item in C.ITEMS:
        for kind, q in queries(item):
            if args.dry_run:
                print(f"{item:6s} {kind:15s} {q}")
                continue
            for ix in INDEXES:
                r = ask(ix, q)
                rows.append(dict(item=item, index=ix, kind=kind, query=q, **r))
                time.sleep(0.2)
            print(item, kind, [rr["count"] for rr in rows[-len(INDEXES):]], flush=True)
    if args.dry_run:
        return
    cnt = pd.DataFrame(rows)
    cnt.to_csv(os.path.join(C.OUT, "c10_exposure_counts.csv"), index=False)
    analyse(cnt)


def analyse(cnt):
    c9 = json.load(open(os.path.join(C.OUT, "c9_item_properties.json")))
    shift = pd.DataFrame(c9["main"]["items"]).set_index("item")["shift"]
    # primary exposure (ICAR-specific): the stem as administered; for the two odd-one-out items, whose stems
    # are the generic ICAR instruction, the co-occurrence of their six options instead
    prim = cnt[((cnt.kind == "stem") & ~cnt.item.isin(ODD_ONE_OUT)) | (cnt.kind == "options_and")]
    # secondary exposure (content): also the bare series, which can occur outside ICAR (squares, Lucas numbers)
    sec = cnt[~((cnt.kind == "stem") & cnt.item.isin(ODD_ONE_OUT))]
    res = {}
    for label, sub in (("primary_stem", prim), ("secondary_content", sec)):
        ex = sub.groupby(["item", "index"])["count"].max().unstack()
        res[label] = analyse_one(ex, shift)
    C.save_json(res, "c10_exposure.json")
    print(json.dumps(res, indent=1, default=str))


def analyse_one(ex, shift):
    res = {}
    rng = np.random.default_rng(10102026)

    def corr(x, y, nperm=20000):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = ~(np.isnan(x) | np.isnan(y))
        x, y = x[ok], y[ok]
        if np.ptp(x) == 0:
            return dict(rho=None, perm_p=None, n=int(ok.sum()), note="no variation in exposure")
        r = spearmanr(x, y).statistic
        perm = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(nperm)])
        return dict(rho=float(r), perm_p=float((1 + np.sum(np.abs(perm) >= abs(r) - 1e-12)) / (nperm + 1)), n=int(ok.sum()))

    expo_any = np.log1p(ex.max(axis=1))
    res["pooled_any_index"] = corr(expo_any.reindex(shift.index), shift)
    res["per_index"] = {ix: corr(np.log1p(ex[ix]).reindex(shift.index), shift) for ix in ex.columns}
    res["items_exposed"] = {ix: sorted(ex.index[ex[ix] > 0].tolist()) for ix in ex.columns}
    # the two OLMo-2 models against their own training data
    d = C.load_long(os.path.join(C.ROOT, "data", "responses_free25.jsonl"))
    models = sorted(d.model.unique())
    P, _, _ = C.model_props(d, models)
    TH = C.loo_theta(P)
    E = 1 / (1 + np.exp(-C.A * (TH - C.B)))
    olmo = [k for k, m in enumerate(models) if "OLMo-2" in m]
    resid = np.nanmean((P - E)[olmo], 0)                      # mean over the OLMo-2 models
    res["olmo_own_data"] = corr(np.log1p(ex[OLMO_INDEX]).reindex(C.ITEMS).to_numpy(), resid)
    res["olmo_models"] = [models[k] for k in olmo]
    return res


if __name__ == "__main__":
    main()
