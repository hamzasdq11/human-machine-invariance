"""Integrity audit of the machine responses, and a lenient re-parse.

Checks, row by row: the logged option order equals the deterministic permutation
of (model, item, variant, seed); the logged presented key equals the key's position
under that order; re-parsing the raw text with the administration parser gives the
logged choice; and `correct` equals (choice == presented key).

Lenient re-parse (sensitivity only): for rows the strict parser left missing, take
the option whose text appears EARLIEST in the response. Writes
data/responses_free25_lenient.jsonl (and _temp0_lenient), scored outcomes only; those
files are distributed, so the lenient sensitivity analysis runs without this script.
"""
import os, sys, json, random, hashlib, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, numpy as np
from scripts.admin_parser import parse_free_response, _norm

# Needs material that is not distributed: the ICAR item text (PsychArchives Scientific Use Licence;
# set ICAR_ITEMS, see data/README.md) and the full response logs with model output text and option
# orders (FULL_LOGS_DIR; available from the author to researchers who hold the PsychArchives licence).
_ITEMS = os.environ.get("ICAR_ITEMS"); FULL = os.environ.get("FULL_LOGS_DIR")
if not (_ITEMS and FULL):
    sys.exit("audit_scoring25 needs ICAR_ITEMS and FULL_LOGS_DIR (see data/README.md); the public "
             "response files carry only scored outcomes.")
ITEMS = {d["item_id"]: d for d in json.load(open(_ITEMS))}

def perm(d, key):
    rnd = random.Random(hashlib.md5("|".join(map(str, key)).encode()).hexdigest())
    k = len(d["options"]) - d["n_fixed_tail"]; head = list(range(k)); rnd.shuffle(head)
    return head + list(range(k, len(d["options"])))

def earliest(text, shown):
    t = " " + _norm(text) + " "; best = None
    for i, o in enumerate(shown):
        o = _norm(o)
        if not o: continue
        m = re.search(r"(?<![a-z0-9])" + re.escape(o) + r"(?![a-z0-9])", t)
        if m and (best is None or m.start() < best[0]): best = (m.start(), i + 1)
    return None if best is None else best[1]

def audit(path):
    df = pd.read_json(path, lines=True); bad = dict(order=0, key=0, choice=0, correct=0)
    len_choice = []; contains_key = 0
    for r in df.itertuples():
        d = ITEMS[r.item_id]
        o = perm(d, (r.model, r.item_id, r.prompt_variant, r.seed))
        bad["order"] += o != list(r.option_order)
        nk = o.index(d["key"] - 1) + 1; bad["key"] += nk != r.presented_key
        shown = [d["options"][i] for i in r.option_order]
        c = parse_free_response(r.raw, shown)
        logged = None if pd.isna(r.choice) else int(r.choice)
        bad["choice"] += c != logged
        if logged is not None: bad["correct"] += int(logged == r.presented_key) != int(r.correct)
        if logged is None:
            e = earliest(r.raw, shown); len_choice.append(e)
            contains_key += _norm(d["options"][d["key"] - 1]) in _norm(r.raw).split() or \
                            (" " + _norm(d["options"][d["key"] - 1]) + " ") in (" " + _norm(r.raw) + " ")
        else:
            len_choice.append(logged)
    df["choice_lenient"] = len_choice
    return df, bad, contains_key

PUBLIC = ["model", "item_id", "domain", "prompt_variant", "seed", "temperature", "mode", "parsable",
          "correct", "latency_s"]

if __name__ == "__main__":
    out = {}
    for tag, name in (("main", "responses_free25.jsonl"), ("temp0", "responses_free25_temp0.jsonl")):
        df, bad, ck = audit(os.path.join(FULL, name))
        miss = df.parsable == False
        rescued = df.loc[miss, "choice_lenient"].notna().sum()
        out[tag] = dict(rows=len(df), mismatches=bad, unparsable=int(miss.sum()),
                        unparsable_containing_key_text=int(ck), lenient_rescued=int(rescued),
                        lenient_rescued_correct=int((df.loc[miss, "choice_lenient"] == df.loc[miss, "presented_key"]).sum()))
        # lenient sensitivity variant, written with scored outcomes only (no text, orders or keys)
        L = df.copy(); ok = L.choice_lenient.notna()
        L["parsable"] = ok
        L["correct"] = np.where(ok, (L.choice_lenient == L.presented_key).astype(float), np.nan)
        L[PUBLIC].to_json(os.path.join("data", name.replace(".jsonl", "_lenient.jsonl")), orient="records", lines=True)
    json.dump(out, open("results/audit_scoring25.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("the as-incorrect and no-v5 variants come from scripts/make_variants25.py (public data only)")
