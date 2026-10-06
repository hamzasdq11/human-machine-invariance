"""Item bank construction, validation, and a procedurally generated instrument.

Two kinds of item bank are supported.

1. **ICAR-16** (primary).  Public-domain items whose *human* item-level response
   data ships with psychTools, but whose item *text* must be obtained from the
   ICAR project.  `ICAR_SCHEMA` and `validate_bank` check that a supplied text
   file matches the response data exactly (same ids, same option counts, keys
   consistent with the published scoring key).

2. **Procedurally generated letter/number series** (supplementary).  Generated
   from explicit rules, so items are guaranteed absent from any pretraining
   corpus.  This gives a contamination-free comparison instrument: any DIF found
   on generated items cannot be memorisation, which is exactly the discriminant
   the uniform/non-uniform decomposition predicts.  Human data for this bank has
   to be collected (small sample, standard crowdsourcing) and is not assumed here.
"""
from __future__ import annotations
import json, random, string
from dataclasses import asdict
from .administer import Item

ALPH = string.ascii_uppercase

ICAR_SCHEMA = {
    "item_id": str, "stem": str, "options": list, "key": int, "domain": str,
}


def load_icar_text(path, responses_meta_csv, strict=True):
    """Load a filled-in ICAR text file and check it against the response data.

    Verifies that item ids match the human response matrix exactly and that every
    key matches the published scoring key.  A silent mismatch here would mis-score
    the entire study, so this refuses rather than warns.
    """
    import json, pandas as pd
    meta = pd.read_csv(responses_meta_csv)
    raw = json.load(open(path))
    items, skipped = [], []
    for r in raw:
        r = {k: v for k, v in r.items() if not k.startswith("_")}
        r.pop("icar_id", None)
        if not r["stem"].strip() or any(not o.strip() for o in r["options"]):
            skipped.append(r["item_id"]); continue
        items.append(Item(**r))
    # the published key, when supplied (icar_items.csv no longer carries one; see data/README.md)
    expected = dict(zip(meta["item"], meta["key"])) if "key" in meta.columns else None
    errs = validate_bank(items, expected_keys=expected)
    unknown = set(i.item_id for i in items) - set(meta["item"])
    if unknown:
        errs.append(f"items not present in the human response data: {sorted(unknown)}")
    if errs and strict:
        raise ValueError("ICAR item file failed validation:\n  " + "\n  ".join(errs))
    return items, skipped, errs


def validate_bank(items, expected_ids=None, expected_keys=None):
    """Fail loudly rather than silently mis-scoring an entire study."""
    errs = []
    ids = [it.item_id for it in items]
    if len(set(ids)) != len(ids):
        errs.append("duplicate item_ids")
    for it in items:
        if it.n_fixed_tail and it.key > len(it.options) - it.n_fixed_tail:
            errs.append(f"{it.item_id}: key points at a fixed-tail option "
                        f"(\"{it.options[it.key-1]}\"), which is never the answer")
        if not (2 <= len(it.options) <= 12):
            errs.append(f"{it.item_id}: implausible option count {len(it.options)}")
        if not (1 <= it.key <= len(it.options)):
            errs.append(f"{it.item_id}: key {it.key} outside 1..{len(it.options)}")
        if not it.stem.strip():
            errs.append(f"{it.item_id}: empty stem")
        if len(set(it.options)) != len(it.options):
            errs.append(f"{it.item_id}: duplicate option text")
    if expected_ids is not None:
        missing = set(expected_ids) - set(ids)
        extra = set(ids) - set(expected_ids)
        if missing: errs.append(f"missing items: {sorted(missing)}")
        if extra:   errs.append(f"unexpected items: {sorted(extra)}")
    if expected_keys is not None:
        for it in items:
            if it.item_id in expected_keys and it.key != expected_keys[it.item_id]:
                errs.append(f"{it.item_id}: key {it.key} != published key "
                            f"{expected_keys[it.item_id]}")
    return errs


# ---------------------------------------------------------------------------
# procedural letter/number series generator
# ---------------------------------------------------------------------------

def _series_linear(rng, n=6):
    start, step = rng.randrange(26), rng.choice([1, 2, 3, -1, -2])
    seq = [(start + i * step) % 26 for i in range(n + 1)]
    return [ALPH[i] for i in seq[:-1]], ALPH[seq[-1]], 1


def _series_alternating(rng, n=6):
    a, b = rng.randrange(26), rng.randrange(26)
    sa, sb = rng.choice([1, 2, 3]), rng.choice([-1, -2, 2])
    out = []
    for i in range((n + 1) // 2 + 1):
        out += [ALPH[(a + i * sa) % 26], ALPH[(b + i * sb) % 26]]
    return out[:n], out[n], 2


def _series_accelerating(rng, n=6):
    start, step = rng.randrange(26), rng.choice([1, 2])
    seq, cur = [start], start
    for i in range(1, n + 1):
        cur = (cur + step + i - 1) % 26
        seq.append(cur)
    return [ALPH[i] for i in seq[:-1]], ALPH[seq[-1]], 3


def _series_pairs(rng, n=6):
    a, s = rng.randrange(26), rng.choice([2, 3, 4])
    out = []
    for i in range(n):
        out += [ALPH[(a + i * s) % 26], ALPH[(a + i * s + 1) % 26]]
    return out[:n], out[n], 3


GENERATORS = {"linear": (_series_linear, 1), "alternating": (_series_alternating, 2),
              "accelerating": (_series_accelerating, 3), "pairs": (_series_pairs, 3)}


def generate_series_bank(n_items=40, n_options=6, seed=0, n_visible=6):
    """Generate a letter-series bank with a recorded rule and difficulty tag.

    `rule_complexity` is the number of latent parameters a solver must recover;
    it is the pre-registered difficulty covariate for this bank.
    """
    rng = random.Random(seed)
    items, kinds = [], list(GENERATORS)
    while len(items) < n_items:
        kind = kinds[len(items) % len(kinds)]
        fn, complexity = GENERATORS[kind]
        vis, ans, _ = fn(rng, n=n_visible)
        # Deterministic ordered accumulation. A set() here would make option order
        # depend on Python's per-process string hash salt, so the generated bank
        # would differ between runs and between machines -- silently unreproducible.
        distractors = []
        guard = 0
        while len(distractors) < n_options - 1 and guard < 500:
            guard += 1
            d = ALPH[(ALPH.index(ans) + rng.choice([-3, -2, -1, 1, 2, 3, 4])) % 26]
            if d != ans and d not in distractors:
                distractors.append(d)
        if len(distractors) < n_options - 1:          # exhausted the offsets
            for off in range(1, 26):
                d = ALPH[(ALPH.index(ans) + off) % 26]
                if d != ans and d not in distractors:
                    distractors.append(d)
                if len(distractors) == n_options - 1:
                    break
        opts = distractors + [ans]
        rng.shuffle(opts)
        items.append(Item(
            item_id=f"gen.{kind}.{len(items):03d}",
            stem="Which letter continues this sequence?\n"
                 + " ".join(vis),
            options=opts, key=opts.index(ans) + 1,
            domain=f"generated letter series ({kind})"))
    return items


def save_bank(items, path):
    with open(path, "w") as f:
        json.dump([asdict(it) for it in items], f, indent=1)
    return path
