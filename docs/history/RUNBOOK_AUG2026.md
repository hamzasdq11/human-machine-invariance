> **Historical document (August 2026), lightly edited in October 2026.** Kept for reference on how the
> machine administration was run.
> It describes the earlier 8-item phase and file names that have since changed; its numbers are not
> the paper's. ICAR item text is no longer distributed with this repository (see `DATA_LICENSES.md`
> and `data/README.md`). For the current analyses, see the top-level `README.md` and `RECOVERY.md`.

# Runbook — executing the machine administration

This is the one step that needs a GPU and network access to HuggingFace; it runs
on free-tier Colab, which provides both. Everything before and after this step runs
locally on CPU.

**Your time: ~5 minutes of work, ~60–90 minutes of unattended running.**

---

## Before you begin

You need:

- A Google account (for Colab). Free tier is sufficient — no Colab Pro required.
- `mim_artifact.zip` (a zip of this repository).

You do **not** need: a GPU, a HuggingFace account, an API key, or any paid service.

---

## Step 0 — Open Colab and set the runtime  *(1 min)*

1. Go to **https://colab.research.google.com**
2. `File` → `Upload notebook` → upload `harness/Colab_Administration.ipynb`
   (unzip `mim_artifact.zip` locally first to get at it).
3. **`Runtime` → `Change runtime type` → Hardware accelerator: `T4 GPU` → Save.**

If you skip 3, cell 2 will stop you with an assertion rather than silently running
on CPU for six hours.

---

## Step 1 — Upload the artifact  *(2 min)*

Run the first code cell. It will prompt for a file — choose **`mim_artifact.zip`**
(the zip itself, not the unzipped folder).

Expected output:

```
working dir: /content/mim
artifact OK
```

Then run the install cell. Expected:

```
torch 2.x.x | CUDA True | Tesla T4
```

**If `CUDA False`:** you missed Step 0.3. Fix the runtime and re-run.

---

## Step 2 — Mount Drive  *(1 min)*

Run the Drive cell and approve the permission prompt.

Expected: `responses -> /content/drive/MyDrive/mim/responses.jsonl`

**Why this matters:** Colab free sessions disconnect, sometimes mid-run. With Drive
mounted, reconnecting and re-running Step 6 picks up exactly where it stopped —
already-recorded cells are never re-administered. Without it, a disconnect costs
you the whole run.

If Drive refuses to mount, the notebook falls back to local storage and says so.
That is workable but you should avoid closing the tab.

---

## Step 3 — Verify the ICAR items  *(30 seconds — nothing to fill in)*

**This step no longer requires any manual work.** The 8 text-administrable ICAR
items are already transcribed into the artifact from the official PsychArchives /
ZPID release of the ICAR item set.

Run the cell. It is a gate, not a form:

- every transcribed answer key is checked against the published scoring key as
  shipped in `psychTools` (all 8 match);
- the deterministic series items are independently re-solved from their stems;
- it refuses to continue on any mismatch, because a transcription error would
  mis-score the entire study without ever failing loudly.

Expected output:

```
  [PASS] file validates
  [PASS] all 8 text items present
  [PASS] keys match published psychTools key
  [PASS] series items independently re-solved (3/3)
  [PASS] fixed tail on every item
  [PASS] key never points at fixed tail
  [PASS] 8 options per item

RESULT: PASS

ICAR items ready: 8
```

**One design note.** Each item has 8 options: six substantive alternatives plus
"None of these" and "I don't know". Option order is randomised per administration,
but those last two are **held in positions 7 and 8**. They are not substantive
alternatives, and shuffling them into the middle would change the instrument
relative to how the human sample saw it. This is recorded as clarification D1 in
the pre-registration.

The matrix-reasoning and three-dimensional-rotation items are figural and remain
excluded by the registered scope condition.

## Step 4 — Assemble the item set  *(instant)*

Run it. Expected:

```
32 items (8 ICAR + 24 generated)
per model: 800 administrations = 25 respondents
```

---

## Step 5 — Model pool  *(instant)*

Run it as-is. Expected:

```
11 models; 8 organisations
```

Leave `USE_GATED = False` unless you already have a HuggingFace token and have
accepted the Llama / Gemma / Mistral licences on their model pages. The 11 core
models already satisfy the registered minimum: 3+ families spanning 0.36B to 7B,
which is two orders of magnitude.

**Do not edit this list after seeing any results.** If a model fails to load, let
it fail — the loop records it in `model_failures.txt` and moves on, and failures
are reported in the paper.

---

## Step 6 — Administer  *(60–90 min, unattended)*

Run it and leave it. Per model you will see:

```
=== Qwen/Qwen2.5-1.5B-Instruct ===
    200/800  (0.31s/item)
    400/800  (0.29s/item)
  800 new administrations in 254s
```

Rough expectations on a T4: sub-1B models ~2–4 min, 1–3B ~4–8 min, 7B in 4-bit
~12–18 min.

### If the session disconnects

Reconnect, re-run cells 1–5 (fast, no re-download of items), then re-run Step 6.
It resumes. You can also close the tab and come back later.

### Known snags

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` on a 7B model | T4 has 15 GB | It is already 4-bit; if it persists, set that model's flag to `True` if not already, or drop it — a recorded failure is fine |
| `401` / `gated repo` | licence not accepted | Leave `USE_GATED = False` |
| `bitsandbytes` import error | install ordering | Re-run the install cell, then `Runtime → Restart session`, then re-run from cell 1 |
| Very slow (`>3s/item` on a small model) | running on CPU | Runtime is not set to T4 |
| A model loads but every response is unparsable | chat template mismatch | Leave it; the parse-rate check excludes it and the paper reports it |

---

## Step 7 — Check quality, then export  *(3 min)*

Run the quality cell. Look at the **parse rate first**:

```
parse rate by model:
HuggingFaceTB/SmolLM2-360M-Instruct    0.71
TinyLlama/TinyLlama-1.1B-Chat-v1.0     0.93
Qwen/Qwen2.5-7B-Instruct               1.00
...
EXCLUDED (<0.90): ['HuggingFaceTB/SmolLM2-360M-Instruct']
```

A low parse rate is an instruction-following failure, not low ability. The
pre-registration excludes those models from the primary analysis rather than
scoring their unparsable responses as wrong — scoring them wrong would convert a
formatting problem into an apparent ability difference, which is exactly the
confound the study is about.

Then run the export cell. It downloads two files:

- **`machine_responses.parquet`** — the respondent × item matrix
- **`machine_meta.csv`** — model, prompt variant, seed, parse rate per respondent

**Save both in the repository root.**

---

## What happens next (~10 min, on CPU)

```bash
python3 scripts/run_invariance.py --machine machine_responses.parquet \
                                  --meta machine_meta.csv
make tables paper
```

This produces:

1. The invariance hierarchy and item-level DIF table
2. The uniform / non-uniform decomposition
3. The variance-components analysis justifying the respondent construction
4. The invariance-purified subtest and the recomputed contrast
5. A regenerated `main.pdf` with the Results section filled in

Then the three numbers the paper is built around finally sit side by side:

| quantity | value |
|---|---|
| placebo floor (procedure alone) | 0.020 |
| human demographic subgroups | ~0.45 |
| **machine respondents** | **← this run** |

---

## One thing to keep in mind

The result is not predetermined. If machine DIF lands inside the human-subgroup
band, the honest finding is that machines are no stranger a population than one
demographic group is to another — which contradicts this paper's hypothesis and is
a publishable null the pre-registration commits to reporting as one. Do not adjust
prompts, models, or items to move the number. That commitment is most of what
makes the result worth anything.
