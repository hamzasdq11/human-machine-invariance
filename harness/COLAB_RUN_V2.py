# ============================================================================
#  ADMINISTRATION v2 -- 25 ICAR items, free-response, with a pilot gate.
#  Self-contained: paste and run. Re-upload mim_artifact.zip first (cell 1).
#
#  What changed and why
#  --------------------
#  * 25 items instead of 8. All 16 verbal-reasoning + 9 letter-series items,
#    with reference difficulties from the ICAR website sample (N=24k-39k) linked
#    onto the psychTools metric by mean-mean equating on the 8 common items
#    (shift +0.102 logits, residual SD 0.067 -- tight). The new items span
#    p = 0.24 to 0.96, so the pool is no longer stuck at the instrument's floor.
#
#  * Free response, not option numbers. Models state the answer; we match it
#    back to an option. A bare number given as the answer to an arithmetic item
#    with numeric options now scores as correct instead of missing, and display
#    position has no channel.
#
#  * A PILOT GATE. The previous run committed a whole pool without checking that
#    any of it could do the instrument. It could not, and that cost an entire
#    administration. Ten minutes up front now decides whether a model is usable.
# ============================================================================
import os, json, gc, time, traceback, torch, pandas as pd, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from harness.administer import (HFBackend, administer_free, pilot_gate,
                                parse_free_response, FREE_FRAMES, Item)

def _patched_init(self, model_name, dtype="auto", device_map="auto",
                  max_new_tokens=24, trust_remote_code=False, **mk):
    self.name = model_name
    self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    self.model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map,
        trust_remote_code=trust_remote_code, **mk)
    self.model.eval(); self.max_new_tokens = max_new_tokens; self._torch = torch
HFBackend.__init__ = _patched_init

ITEMS25 = [Item(**{k: v for k, v in r.items() if k in
                   ("item_id","stem","options","key","domain","n_fixed_tail","source")})
           for r in json.load(open(os.environ["ICAR_ITEMS"]))]   # item text: see data/README.md
print(f"{len(ITEMS25)} ICAR items loaded")

OUT_F = OUT.replace("responses.jsonl", "responses_free25.jsonl")
print("writing to", OUT_F)

QUANT = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True)

# Pool spanning capability. The small models are kept -- they are the low end of
# the ability range, which is exactly what the analysis needs. The larger ones
# supply the spread that was missing.
POOL = [
    ("HuggingFaceTB/SmolLM2-360M-Instruct", False),
    ("Qwen/Qwen2.5-0.5B-Instruct",          False),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0",  False),
    ("Qwen/Qwen2.5-1.5B-Instruct",          False),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", False),
    ("Qwen/Qwen2.5-3B-Instruct",            False),
    ("microsoft/Phi-3.5-mini-instruct",     False),
    ("Qwen/Qwen2.5-7B-Instruct",            True),
    ("allenai/OLMo-2-1124-7B-Instruct",     True),
]
PILOT_ONLY = False        # True = just gate every model, administer nothing

report = []
for name, four_bit in POOL:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    try:
        kw = {"quantization_config": QUANT} if four_bit else {}
        if "internlm" in name: kw["trust_remote_code"] = True
        be = HFBackend(name, **kw)

        ok, acc, parse = pilot_gate(be, ITEMS25, lo=0.30, hi=0.85, seeds=(0,))
        report.append(dict(model=name, pilot_acc=acc, pilot_parse=parse, usable=ok))
        if ok and not PILOT_ONLY:
            n = administer_free(be, ITEMS25, out_path=OUT_F, resume=True, batch_size=8)
            print(f"  {n} administrations in {time.time()-t0:.0f}s")
        elif not ok:
            print("  excluded from the pool -- recorded, not hidden")
        del be
    except Exception as e:
        print(f"  FAILED: {e}"); traceback.print_exc()
        report.append(dict(model=name, pilot_acc=None, pilot_parse=None, usable=False))
    gc.collect(); torch.cuda.empty_cache()

print("\n" + "=" * 70)
print(pd.DataFrame(report).round(3).to_string(index=False))
pd.DataFrame(report).to_csv("pilot_report.csv", index=False)

if os.path.exists(OUT_F):
    d = pd.read_json(OUT_F, lines=True)
    d["respondent"] = d.model + "|" + d.prompt_variant + "|" + d.seed.astype(str)
    w = d.pivot_table(index="respondent", columns="item_id", values="correct", aggfunc="first")
    def alpha(X):
        X = pd.DataFrame(X).dropna(); k = X.shape[1]
        if len(X) < 3 or X.sum(axis=1).var(ddof=1) == 0: return float("nan")
        return k/(k-1)*(1 - X.var(ddof=1).sum()/X.sum(axis=1).var(ddof=1))
    print(f"\n  accuracy {np.nanmean(w.values):.3f} | alpha {alpha(w.values):+.3f} "
          f"| respondents {w.shape[0]} | parse {d.parsable.mean():.3f}")
    print("  (humans: acc 0.642, alpha +0.766 | previous arms: alpha -0.173 and +0.002)")
    print("\n  per model:")
    print(d.groupby("model").agg(acc=("correct","mean"), parse=("parsable","mean")).round(3).to_string())
    from google.colab import files
    files.download(OUT_F); files.download("pilot_report.csv")
