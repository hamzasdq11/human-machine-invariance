"""Machine administration harness (Colab / Kaggle / local).

Implements the pre-registered protocol exactly:
  respondent = (model, prompt_variant, seed);  option order randomised and logged;
  a single deterministic parser;  unparsable -> missing, never scored incorrect.

The output is a tidy response table that the local analysis pipeline consumes.
Nothing in this file may be tuned after seeing performance -- see PREREGISTRATION.md.
"""
from __future__ import annotations
import os, json, re, itertools, hashlib, time
from dataclasses import dataclass, asdict
from typing import Sequence

# ---------------------------------------------------------------------------
# 1. Prompt frames -- FIXED BEFORE ADMINISTRATION.  Five paraphrases.
# ---------------------------------------------------------------------------
PROMPT_VARIANTS = {
    "v1": ("Answer the following multiple-choice question.\n\n"
           "{stem}\n\n{options}\n\nRespond with only the number of the correct option."),
    "v2": ("Below is a multiple-choice question. Select the best answer.\n\n"
           "{stem}\n\n{options}\n\nReply with the option number and nothing else."),
    "v3": ("Question:\n{stem}\n\nOptions:\n{options}\n\n"
           "Which option is correct? Give the number only."),
    "v4": ("You are taking a reasoning test. Read the item and choose one option.\n\n"
           "{stem}\n\n{options}\n\nYour answer (number only):"),
    "v5": ("{stem}\n\n{options}\n\nChoose the correct option. Answer with its number."),
}

# ---------------------------------------------------------------------------
# 2. Response parser -- FIXED.  Returns an option index or None.
# ---------------------------------------------------------------------------
# standalone integer: not part of a larger number and not a decimal like "3.5".
# A trailing sentence period is allowed ("The answer is 4.").
_NUM = re.compile(r"(?<!\d)(?<!\.)([1-9]\d?)(?!\d)(?!\.\d)")

def parse_response(text: str, n_options: int):
    """Deterministic extraction of a chosen option number.

    Registered rule: take the FIRST standalone integer in [1, n_options] that
    appears in the response. Anything else is unparsable (coded missing).
    """
    if not text:
        return None
    for m in _NUM.finditer(text.strip()):
        v = int(m.group(1))
        if 1 <= v <= n_options:
            return v
    return None


# ---------------------------------------------------------------------------
# 2b. Free-response matching
# ---------------------------------------------------------------------------
_PUNCT = str.maketrans("", "", ".,;:!?()[]{}\"'`")


def _norm(s):
    return " ".join(str(s).lower().translate(_PUNCT).split())


def parse_free_response(text, options):
    """Match a free-text answer to one of the presented options.

    Asking for the option *number* is ambiguous with the answer itself whenever
    the options are bare digits or single letters -- a model that replies with the
    correct number to an arithmetic item is correct, and a position parser records
    it as missing. Here the model states the answer and we match it back, so the
    reporting step cannot fail in that way.

    Registered rule, in order: exact normalised match; then the answer appearing
    as a standalone token/phrase in the response; then a unique prefix match.
    Ambiguity between two or more options is unparsable, never a guess.
    """
    if not text:
        return None
    t = _norm(text)
    if not t:
        return None
    opts = [_norm(o) for o in options]

    for i, o in enumerate(opts):                      # exact
        if t == o:
            return i + 1

    toks = t.split()
    hits = []
    for i, o in enumerate(opts):                      # contained as a whole phrase
        ow = o.split()
        if len(ow) == 1:
            if o in toks:
                hits.append(i + 1)
        else:
            for j in range(len(toks) - len(ow) + 1):
                if toks[j:j + len(ow)] == ow:
                    hits.append(i + 1); break
    if len(set(hits)) == 1:
        return hits[0]
    if len(set(hits)) > 1:
        return None                                   # ambiguous -> missing

    hits = [i + 1 for i, o in enumerate(opts) if o and t.startswith(o)]
    return hits[0] if len(hits) == 1 else None


FREE_FRAMES = {
    "v1": ("Answer the following question.\n\n{stem}\n\nThe possible answers are:\n{options}\n\n"
           "Reply with the answer itself, exactly as written above, and nothing else."),
    "v2": ("Below is a question with a list of possible answers.\n\n{stem}\n\n{options}\n\n"
           "State the correct answer exactly as it appears in the list. Nothing else."),
    "v3": ("Question:\n{stem}\n\nPossible answers:\n{options}\n\n"
           "Which is correct? Reply with the answer text only."),
    "v4": ("You are taking a reasoning test.\n\n{stem}\n\nChoose from:\n{options}\n\n"
           "Write the correct answer exactly as listed, and nothing more."),
    "v5": "{stem}\n\n{options}\n\nAnswer (copy the correct option exactly):",
}


# ---------------------------------------------------------------------------
# 3. Item representation
# ---------------------------------------------------------------------------
@dataclass
class Item:
    item_id: str
    stem: str
    options: Sequence[str]
    key: int                 # 1-based index of the correct option
    domain: str = ""
    n_fixed_tail: int = 0    # trailing options held in place (e.g. "None of these",
                             # "I don't know") -- these are not substantive
                             # alternatives, and shuffling them into the middle would
                             # change the instrument relative to how humans saw it
    source: str = ""

    def permutation(self, rnd):
        """A random option order that respects the fixed tail."""
        n = len(self.options)
        k = n - self.n_fixed_tail
        head = list(range(k))
        rnd.shuffle(head)
        return head + list(range(k, n))

    def render(self, order):
        """Render with a given permutation of options; return text and new key."""
        opts = [self.options[i] for i in order]
        body = "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
        new_key = order.index(self.key - 1) + 1
        return body, new_key

    def render_plain(self, order):
        """Options as an unnumbered list, for free-response administration."""
        opts = [self.options[i] for i in order]
        body = "\n".join(f"- {o}" for o in opts)
        return body, order.index(self.key - 1) + 1, opts


def load_items(path):
    with open(path) as f:
        raw = json.load(f)
    return [Item(**r) for r in raw]


# ---------------------------------------------------------------------------
# 4. Backends
# ---------------------------------------------------------------------------
class HFBackend:
    """transformers backend; works on Colab GPU or local CPU."""

    def __init__(self, model_name, dtype="auto", device_map="auto", max_new_tokens=16,
                 trust_remote_code=False, **model_kwargs):
        """Extra keyword arguments (e.g. quantization_config) pass through to
        from_pretrained, so 4-bit loading needs no separate code path."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map,
            trust_remote_code=trust_remote_code, **model_kwargs)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def _wrap(self, prompt):
        msgs = [{"role": "user", "content": prompt}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
        except Exception:
            return prompt

    def generate_batch(self, prompts, seeds, temperature=0.7):
        """Generate for a batch of prompts. ~5-10x faster than one at a time.

        The seed is set once per batch from the first element's seed; sampling
        randomness therefore depends on (seed, batch composition), and batch
        composition is a deterministic function of the pending cell order. Runs
        are reproducible given the same resume state and batch size, which is
        recorded with every response.
        """
        torch = self._torch
        torch.manual_seed(int(seeds[0]))
        texts = [self._wrap(p) for p in prompts]
        pad_side = self.tok.padding_side
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                top_p=1.0, pad_token_id=self.tok.pad_token_id)
        self.tok.padding_side = pad_side
        n_in = enc["input_ids"].shape[1]
        return [self.tok.decode(o[n_in:], skip_special_tokens=True) for o in out]


    # ---------------------------------------------------------------- log-prob
    def score_options(self, context, options):
        """Log-likelihood of each option's TEXT as a continuation of `context`.

        Asking a model to reply with the *number* of an option conflates two
        things: whether it knows the answer, and whether it reports positions
        the way the scorer expects. On items whose options are bare digits or
        single letters those are not separable -- a model that answers an
        arithmetic item with the correct number is correct, and a position-parser
        records it as missing. Scoring the option text directly removes the reporting
        step entirely.

        Returns (mean_logprob_per_token, total_logprob) for each option.
        """
        torch = self._torch
        ctx_ids = self.tok(context, return_tensors="pt").input_ids
        n_ctx = ctx_ids.shape[1]
        means, totals = [], []
        for opt in options:
            full = self.tok(context + " " + str(opt), return_tensors="pt").input_ids
            full = full.to(self.model.device)
            with torch.no_grad():
                logits = self.model(full).logits
            lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
            tgt = full[0, 1:]
            tok_lp = lp[torch.arange(len(tgt)), tgt][n_ctx - 1:]
            totals.append(float(tok_lp.sum()))
            means.append(float(tok_lp.mean()) if len(tok_lp) else float("-inf"))
        return means, totals

    def generate(self, prompt, seed, temperature=0.7):
        torch = self._torch
        torch.manual_seed(seed)
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                top_p=1.0, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 5. Administration loop
# ---------------------------------------------------------------------------
def administer(backend, items, prompt_variants=None, seeds=(0, 1, 2, 3, 4),
               temperature=0.7, out_path="responses.jsonl", resume=True,
               batch_size=1, progress_every=200):
    """Administer every item under every (prompt variant, seed) combination.

    batch_size > 1 uses the backend's generate_batch when available; this is the
    difference between a ~4 hour and a ~40 minute run over a full model pool.
    """
    import random
    pv = prompt_variants or list(PROMPT_VARIANTS)
    done = set()
    if resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line); done.add((r["model"], r["item_id"], r["prompt_variant"], r["seed"]))
                except Exception:
                    pass
    # Build the pending work list first so batching and progress are meaningful.
    pending = []
    for it, var, seed in itertools.product(items, pv, seeds):
        key = (backend.name, it.item_id, var, seed)
        if key in done:
            continue
        # option order is a deterministic function of (item, variant, seed):
        # reproducible, and logged either way.
        rnd = random.Random(hashlib.md5("|".join(map(str, key)).encode()).hexdigest())
        order = it.permutation(rnd)
        body, new_key = it.render(order)
        prompt = PROMPT_VARIANTS[var].format(stem=it.stem, options=body)
        pending.append((it, var, seed, order, new_key, prompt))

    can_batch = batch_size > 1 and hasattr(backend, "generate_batch")
    n_done = 0
    with open(out_path, "a") as f:
        for start in range(0, len(pending), batch_size if can_batch else 1):
            chunk = pending[start:start + (batch_size if can_batch else 1)]
            t0 = time.time()
            if can_batch:
                try:
                    raws = backend.generate_batch([c[5] for c in chunk],
                                                  [c[2] for c in chunk], temperature)
                    errs = [None] * len(chunk)
                except Exception as e:
                    raws, errs = [""] * len(chunk), [repr(e)] * len(chunk)
            else:
                raws, errs = [], []
                for c in chunk:
                    try:
                        raws.append(backend.generate(c[5], seed=c[2], temperature=temperature))
                        errs.append(None)
                    except Exception as e:
                        raws.append(""); errs.append(repr(e))
            dt = round((time.time() - t0) / max(len(chunk), 1), 3)
            for (it, var, seed, order, new_key, _), raw, err in zip(chunk, raws, errs):
                choice = parse_response(raw, len(it.options))
                rec = dict(model=backend.name, item_id=it.item_id, domain=it.domain,
                           prompt_variant=var, seed=seed, temperature=temperature,
                           option_order=order, presented_key=new_key,
                           raw=raw[:400], choice=choice,
                           correct=None if choice is None else int(choice == new_key),
                           parsable=choice is not None, error=err,
                           batch_size=len(chunk), latency_s=dt)
                f.write(json.dumps(rec) + "\n")
                n_done += 1
            f.flush()
            if progress_every and n_done % progress_every < len(chunk):
                print(f"    {n_done}/{len(pending)}  ({dt:.2f}s/item)", flush=True)
    return n_done


# Same instruction frames, but the model is not asked to emit a position; the
# option text is scored directly, so these end at the point a human would read
# the alternatives.
LOGPROB_FRAMES = {
    "v1": "Answer the following multiple-choice question.\n\n{stem}\n\n{options}\n\nAnswer:",
    "v2": "Below is a multiple-choice question. Select the best answer.\n\n{stem}\n\n{options}\n\nAnswer:",
    "v3": "Question:\n{stem}\n\nOptions:\n{options}\n\nAnswer:",
    "v4": "You are taking a reasoning test. Read the item and choose one option.\n\n{stem}\n\n{options}\n\nAnswer:",
    "v5": "{stem}\n\n{options}\n\nAnswer:",
}


def administer_logprob(backend, items, prompt_variants=None, orders=(0, 1, 2, 3, 4),
                       out_path="responses_logprob.jsonl", resume=True,
                       norm="mean", progress_every=100):
    """Administer by scoring option text rather than by generating a position.

    `orders` replaces `seeds`: scoring is deterministic, so the only per-
    respondent variation left is which option permutation was presented. Keeping
    five of them preserves the respondent structure and still lets position
    effects be measured rather than assumed away.
    """
    import random
    pv = prompt_variants or list(LOGPROB_FRAMES)
    done = set()
    if resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["model"], r["item_id"], r["prompt_variant"], r["seed"]))
                except Exception:
                    pass

    pending = []
    for it, var, o in itertools.product(items, pv, orders):
        key = (backend.name, it.item_id, var, o)
        if key in done:
            continue
        rnd = random.Random(hashlib.md5("|".join(map(str, key)).encode()).hexdigest())
        order = it.permutation(rnd)
        body, new_key = it.render(order)
        ctx = LOGPROB_FRAMES[var].format(stem=it.stem, options=body)
        shown = [it.options[k] for k in order]
        pending.append((it, var, o, order, new_key, ctx, shown))

    n_done = 0
    with open(out_path, "a") as f:
        for it, var, o, order, new_key, ctx, shown in pending:
            t0 = time.time()
            try:
                means, totals = backend.score_options(ctx, shown)
                err = None
            except Exception as e:
                means = totals = [float("-inf")] * len(shown)
                err = repr(e)
            score = means if norm == "mean" else totals
            choice = int(max(range(len(score)), key=lambda i: score[i])) + 1
            rec = dict(model=backend.name, item_id=it.item_id, domain=it.domain,
                       prompt_variant=var, seed=o, temperature=None,
                       option_order=order, presented_key=new_key,
                       raw="", choice=choice,
                       correct=None if err else int(choice == new_key),
                       parsable=err is None, error=err,
                       logprob_mean=means, logprob_total=totals, scoring=norm,
                       latency_s=round(time.time() - t0, 3))
            f.write(json.dumps(rec) + "\n")
            n_done += 1
            if progress_every and n_done % progress_every == 0:
                print(f"    {n_done}/{len(pending)}", flush=True)
            f.flush()
    return n_done


def administer_free(backend, items, prompt_variants=None, seeds=(0, 1, 2, 3, 4),
                    temperature=0.7, out_path="responses_free.jsonl", resume=True,
                    batch_size=8, max_new_tokens=24, progress_every=200):
    """Administer by free response, matching the answer text back to an option."""
    import random
    pv = prompt_variants or list(FREE_FRAMES)
    done = set()
    if resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["model"], r["item_id"], r["prompt_variant"], r["seed"]))
                except Exception:
                    pass
    old_max = getattr(backend, "max_new_tokens", None)
    if old_max is not None:
        backend.max_new_tokens = max_new_tokens     # room to name a phrase answer

    pending = []
    for it, var, seed in itertools.product(items, pv, seeds):
        key = (backend.name, it.item_id, var, seed)
        if key in done:
            continue
        rnd = random.Random(hashlib.md5("|".join(map(str, key)).encode()).hexdigest())
        order = it.permutation(rnd)
        body, new_key, shown = it.render_plain(order)
        pending.append((it, var, seed, order, new_key, shown,
                        FREE_FRAMES[var].format(stem=it.stem, options=body)))

    can_batch = batch_size > 1 and hasattr(backend, "generate_batch")
    n = 0
    with open(out_path, "a") as f:
        step = batch_size if can_batch else 1
        for st in range(0, len(pending), step):
            chunk = pending[st:st + step]
            t0 = time.time()
            if can_batch:
                try:
                    raws = backend.generate_batch([c[6] for c in chunk], [c[2] for c in chunk], temperature)
                    errs = [None] * len(chunk)
                except Exception as e:
                    raws, errs = [""] * len(chunk), [repr(e)] * len(chunk)
            else:
                raws, errs = [], []
                for c in chunk:
                    try:
                        raws.append(backend.generate(c[6], seed=c[2], temperature=temperature)); errs.append(None)
                    except Exception as e:
                        raws.append(""); errs.append(repr(e))
            dt = round((time.time() - t0) / max(len(chunk), 1), 3)
            for (it, var, seed, order, new_key, shown, _), raw, err in zip(chunk, raws, errs):
                ch = parse_free_response(raw, shown)
                f.write(json.dumps(dict(
                    model=backend.name, item_id=it.item_id, domain=it.domain,
                    prompt_variant=var, seed=seed, temperature=temperature,
                    option_order=order, presented_key=new_key, raw=raw[:300], choice=ch,
                    correct=None if ch is None else int(ch == new_key),
                    parsable=ch is not None, error=err, mode="free",
                    latency_s=dt)) + "\n")
                n += 1
            f.flush()
            if progress_every and n % progress_every < len(chunk):
                print(f"    {n}/{len(pending)}", flush=True)
    if old_max is not None:
        backend.max_new_tokens = old_max
    return n


def pilot_gate(backend, items, lo=0.30, hi=0.85, n_items=None, seeds=(0, 1),
               out_path=None, verbose=True):
    """Check a model lands mid-range on the instrument BEFORE committing a pool.

    The failure that cost this study a full administration was a pool sitting at
    the instrument's floor: with no ability variance there is no latent trait to
    place on a common scale, and no amount of downstream analysis recovers it.
    A ten-minute check would have caught it, so it is now a gate.
    """
    import tempfile
    sub = items if n_items is None else items[:n_items]
    tmp = out_path or os.path.join(tempfile.gettempdir(), f"pilot_{abs(hash(backend.name))}.jsonl")
    administer_free(backend, sub, seeds=seeds, out_path=tmp, resume=False, progress_every=0)
    import pandas as pd
    d = pd.read_json(tmp, lines=True)
    acc, parse = d.correct.mean(), d.parsable.mean()
    ok = (lo <= acc <= hi) and parse >= 0.80
    if verbose:
        print(f"  PILOT {backend.name}: accuracy {acc:.3f}  parse {parse:.3f}  "
              f"-> {'USABLE' if ok else 'REJECT'} (want {lo}-{hi}, parse>=0.80)")
        if acc < lo:  print("    at/near FLOOR: contributes no ability variance")
        if acc > hi:  print("    at/near CEILING: same problem, inverted")
    return ok, float(acc), float(parse)


def to_response_matrix(jsonl_path, item_order=None):
    """Collapse the tidy log into the (respondent x item) matrix the pipeline wants.

    One respondent = one (model, prompt_variant, seed) triple, as registered.
    Unparsable responses become NaN.
    """
    import pandas as pd, numpy as np
    df = pd.read_json(jsonl_path, lines=True)
    df["respondent"] = df["model"] + "|" + df["prompt_variant"] + "|" + df["seed"].astype(str)
    wide = df.pivot_table(index="respondent", columns="item_id", values="correct",
                          aggfunc="first")
    if item_order is not None:
        wide = wide.reindex(columns=list(item_order))
    meta = df.groupby("respondent").agg(model=("model", "first"),
                                        prompt_variant=("prompt_variant", "first"),
                                        seed=("seed", "first"),
                                        parsable_rate=("parsable", "mean"))
    return wide, meta
