"""Sensitivity variants of the response files that need only the public data.

    data/responses_free25_asincorrect.jsonl   unparsable answers counted as wrong
    data/responses_free25_nov5.jsonl          frame v5 dropped
(and the same for _temp0). The lenient-parser variant needs the model output text and is
distributed ready-made (data/responses_free25_lenient.jsonl; see scripts/audit_scoring25.py).
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
for name in ("responses_free25.jsonl", "responses_free25_temp0.jsonl"):
    path = os.path.join(ROOT, "data", name)
    df = pd.read_json(path, lines=True)
    W = df.copy()
    W["correct"] = np.where(W.parsable, W.correct, 0.0); W["parsable"] = True
    W.to_json(path.replace(".jsonl", "_asincorrect.jsonl"), orient="records", lines=True)
    df[df.prompt_variant != "v5"].to_json(path.replace(".jsonl", "_nov5.jsonl"), orient="records", lines=True)
    print(name, len(df), "->", "asincorrect", len(W), "nov5", int((df.prompt_variant != "v5").sum()))
