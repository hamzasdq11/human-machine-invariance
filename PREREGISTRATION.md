# Pre-registration
## Measurement Invariance Between Human and Machine Respondents on Cognitive Assessments

**Version 1.0 · 18 August 2026 · registered before inspection of any human-vs-machine result**

> **Note on this public copy (added 7 October 2026).** The document was not deposited with a
> registry, so its date rests on the author's own record. One phrase in D3 that quoted an ICAR item
> and its answer is redacted, because ICAR item text is distributed under a scientific-use licence
> (see `DATA_LICENSES.md`). Nothing else has been changed. Deviations after D5 (D6–D16) are listed in
> the paper's Appendix B.

This document fixes the hypotheses, administration protocol, analysis plan, decision
rules and stopping rules *before* any model has been administered any item. It is
written so that a reader can determine, after the fact, which analyses were planned
and which were exploratory. Deviations will be reported in a numbered addendum with
dates, not silently folded into the method section.

---

### 1. Background and motivation

Claims of the form *"model M achieves human-level performance on cognitive test T"*
place human and machine respondents on a common scale and compare their positions.
Such a comparison is interpretable only if T measures the same latent construct, with
the same metric, in both populations — the psychometric condition of **measurement
invariance**. Its violation is **differential item functioning (DIF)**. To our knowledge
this condition has never been tested for any human–machine comparison, although the
field's own systematic review of LLM psychometrics names construct equivalence as an
unresolved foundational question.

### 2. Design

Two-group anchored DIF, reference group = humans, focal group = machine respondents.

**Why anchored.** Recent simulation work shows that IRT estimation fails in the AI
benchmarking regime (few respondents, many items): marginal ML estimation is reported
to fail in a majority of conditions and item-parameter recovery is unreliable below
roughly 100 respondents. The anchored design does not enter that regime: item
parameters are calibrated once on the human sample, where N is in the thousands, and
the machine group contributes only (i) a two-parameter latent distribution identified
from anchor items and (ii) one studied item at a time. Consistency therefore depends
on the *human* sample size. This is a design decision, registered in advance, not a
post-hoc defence.

### 3. Confirmatory hypotheses

| ID | Hypothesis | Primary test | Falsified if |
|----|------------|--------------|--------------|
| **H1** | DIF prevalence between humans and machines exceeds both the random-split placebo rate and the human-subgroup rate on the same instrument | Two-proportion comparison of BH-corrected flag rates, bootstrap CI on the difference | CI on the difference includes 0 |
| **H1a** | DIF is systematic: item demand features predict signed DIF | Held-out-*instrument* cross-validated R² of the feature regression | Held-out R² ≤ 0.10 |
| **H1b** | At least one published comparative conclusion reverses or loses significance on the invariance-purified subtest | Recomputation of the published contrast on I\* with bootstrap CI | No contrast changes direction or significance status |
| **H1c** | (null competitor) Apparent DIF is an artefact of ability-distribution separation | Matched-ability subsampling; separation-envelope simulation | Effect vanishes under matching, or observed separation lies outside the validated envelope |
| **H4a/b** | Contamination produces predominantly *uniform* DIF; cognitive divergence produces *non-uniform* DIF | AUC of the uniformity index UI = LR_uniform / LR_total discriminating post-cutoff from pre-cutoff items | AUC ≤ 0.60 |

**H1 is the primary hypothesis.** All others are secondary and will be reported as such.

### 4. Instruments

*Primary.* ICAR-16 (International Cognitive Ability Resource sample set; public
domain), N = 1,525 human respondents, 16 items in four domains: verbal reasoning,
letter/number series, matrix reasoning, three-dimensional rotation.

**Registered scope condition.** Only the verbal-reasoning and letter/number-series
domains (8 items) are administrable to a text-only model. Matrix-reasoning and
rotation items are figural. Text-only models will be administered the 8 text items;
vision-language models, where included, will be administered all 16. Analyses are
pre-specified separately for the text-only subset and the full set; the text-only
subset is primary. This restriction is registered in advance precisely because
discovering it after seeing results would be an obvious researcher degree of freedom.

*Secondary.* H-ARC (1,729 participants × 800 ARC tasks, with action traces) for
replication and for the error-profile analysis.

*Calibration references.* BFI (N = 2,800) and SPI (N = 4,000), both with demographic
variables, supply the human-subgroup DIF baseline. Polytomous items are dichotomised
at the pooled-sample median, so the split point cannot itself induce group differences.

### 5. Machine administration protocol (fixed in advance)

- **Models.** ≥ 12 open-weight models spanning ≥ 3 families and ≥ 2 orders of magnitude
  in parameter count. The list is fixed before administration and reported in full,
  including any model that fails to produce parsable responses.
- **Respondent definition.** One focal-group respondent = one (model, prompt-variant,
  seed) triple. Non-independence of these respondents is quantified by a variance-
  components analysis reported in the methods, not assumed away.
- **Prompt variants.** 5 paraphrases of a fixed instruction frame, written before
  administration and released verbatim.
- **Seeds.** 5 per (model, prompt variant); temperature fixed at 0.7 for the main
  analysis, with a temperature-0 arm as a registered robustness check.
- **Option order.** Randomised per administration with the permutation recorded.
- **Response extraction.** A single deterministic parser fixed in advance. Unparsable
  responses are coded missing, never scored incorrect. The unparsable rate is reported
  per model; a model exceeding 10% unparsable is reported separately and excluded from
  the primary analysis.
- **No prompt tuning.** Prompts will not be revised after observing performance.

### 6. Analysis plan

1. Calibrate 2PL item parameters on the human sample (θ ~ N(0,1) for identification).
2. Identify the machine group's latent distribution N(μ, σ²) from anchor items.
3. Item-by-item likelihood-ratio DIF with anchors fixed:
   total (df = 2), uniform (df = 1), non-uniform (df = 1).
4. Iterative anchor purification to a stable set, minimum 4 anchors.
5. Benjamini–Hochberg control at α = 0.05 across items, within instrument.
6. Effect sizes: Raju signed and unsigned area; Mantel-Haenszel Δ with ETS A/B/C.
7. Secondary estimator: Mantel-Haenszel. **Conclusions that differ between the two
   estimators will be reported as unresolved, not adjudicated post hoc.**
8. Feature regression of signed DIF on demand annotations, validated on held-out
   instruments.
9. Purified-subtest recomputation of published comparative contrasts.

### 7. Mandatory controls

| Control | Purpose | Pre-specified expectation |
|---|---|---|
| Random-split placebo | Bounds the pipeline's false-positive floor on real data | Flag rate ≈ nominal α |
| Human-subgroup DIF | Establishes ordinary cross-population non-invariance | Reported, not predicted |
| Model-vs-model DIF | Tests whether machines are one population | Reported, not predicted |
| Matched-ability subsampling | H1c | Effect persists |
| Separation-envelope simulation | Validates the estimator at the observed group separation | Observed separation inside envelope |
| Temporal-cutoff partition | Contamination ground truth | Uniform DIF concentrated pre-cutoff |

### 8. Stopping and abandonment rules

Phase 0 (ICAR, 6 models) is run first and its result is binding:

- **Placebo failure** — random human splits show a flag rate materially above nominal:
  halt, repair the pipeline, re-register.
- **Null result** — human-vs-machine DIF is comparable to human-subgroup DIF: H1 is
  false. This is a publishable null and will be reported as one; it will not be
  rescued by switching instruments or estimators.
- **Estimator disagreement** — LR and MH disagree on the primary claim: reported as
  unresolved.
- **Out-of-envelope separation** — observed |μ| beyond the validated range: the
  primary claim is withdrawn and only the envelope-valid subset is reported.

### 9. What would *not* change the plan

Moderate effect sizes; disagreement between instruments; failure of the optional
ability-space (configural) or theory-of-mind sections; appearance of a related paper.
None of these licenses changing the primary hypothesis, the instrument set, the
estimator, or the α level.

### 10. Deviations and clarifications

Numbered and dated as they occur.

**D1 — 18 August 2026 — clarification, pre-administration.** Each ICAR item
presents eight options: six substantive alternatives plus "None of these" and
"I don't know". §5 registered option-order randomisation without specifying its
scope. Clarified before any administration: randomisation applies to the six
substantive options only; the final two are held in positions 7 and 8. They are
not substantive alternatives, and moving them would alter the instrument relative
to the administration the human reference sample received. No result had been
observed when this was fixed.

**D5 — 19 August 2026 — item pool extended to 25, pre-analysis.** §4 registered the
8 text-administrable ICAR-16 items. D4 showed that pool sits at the machine floor,
partly because all 8 are mid-to-hard for humans (p = 0.47–0.74). The archived ICAR
release supplies text and published scoring keys for all 16 verbal-reasoning and 9
letter-series items, spanning p = 0.24–0.96. Reference difficulties for the 17 new
items come from the ICAR website sample (N = 24k–39k), linked onto the psychTools
metric by mean–mean equating on the 8 common items: shift +0.102 logits, residual
SD 0.067, max |residual| 0.105. All 25 transcribed keys match the published scoring
key. Primary analysis remains 2PL on the 8 items with item-level data; the 25-item
set is a 1PL extension, reported separately. The ZPID aggregates are **not** mixed
into the anchor (see D4).

**D4 — 19 August 2026 — focal pool at instrument floor; pilot gate added.**
Both administrations left the machine population at the instrument's floor: 7 of 8
items at or below chance, only `reason.16` solved (0.675 vs 0.123 elsewhere), and
only 2 of 8 models significantly above chance. With no ability variance there is no
latent dimension to place on the human scale, so invariance is **not estimable** —
which is distinct from invariance failing, and is reported as such. Simulation on
the real human calibration confirms the diagnosis: a floor-only pool yields α ≈
0.07 (observed: +0.002), and adding two capable respondents raises it to +0.84.

Cause: the pool was registered for openness and reproducibility without checking
that any of its models could perform the instrument. Remedy, registered before any
invariance statistic is computed: (i) a **pilot gate** requiring 0.30 ≤ accuracy ≤
0.85 and parse rate ≥ 0.80 before a model enters the pool, with every gated model
reported; (ii) a pool spanning capability rather than uniformly small.

**D3 — 18 August 2026 — response format, post-administration, pre-analysis.**
The generation-based administration registered in §5 asked models to reply with
the *number* of the chosen option. On items whose options are bare digits or
single letters this is not separable from the answer itself: models replying
with the correct number to an arithmetic item [item text and answer redacted on
7 October 2026; see the note at the top] were correct and were recorded as
missing. What survived was dominated by display-position preference (one model
selected position 1 on 56% of items; whether the answer fell in a model's
preferred position roughly doubled P(correct), z = 6.5). The resulting response
matrix had Cronbach alpha = -0.173 against the human sample's +0.766, with every
item-total correlation at or below zero: no common latent dimension, hence
nothing to place on a common scale.

This was detected at the response-quality stage specified in §5, **before any DIF
statistic was computed**, and no invariance result had been inspected when the
decision was taken.

Superseded in part by D4: log-probability scoring over all 8 options placed 64.4%
of choices on "None of these" / "I don't know" — short fluent strings a likelihood
scorer favours regardless of content. Re-scoring on the six substantive options
removes this (α +0.002 mean, −0.072 total; both normalisations agree). The final
administration (D5) uses free response matched back to an option, which avoids both
the position-report ambiguity and the fluency bias.

Action: re-administer by scoring option *text* log-likelihood, with items,
prompts, permutation scheme and respondent structure otherwise unchanged. The
model is not asked to report a position, so neither the reporting ambiguity nor
position preference has a channel.

Registered before the re-run: on simulated respondents of known ability the
log-probability path recovers alpha = 0.751 against the human sample's 0.766,
while a pure position-preference responder yields 0.050, and a population with
the narrow ability spread of the real model pool yields 0.291. The pipeline is
therefore capable of measuring a trait, which is what makes the re-run's outcome
interpretable either way (`tests/test_logprob_recovers_trait.py`).

Both arms are reported. If the log-probability arm is also incoherent, the
incoherence is a property of the models rather than of the response format, and
that is the result. If it is coherent, the generation arm was an administration
artifact and is reported as such. Neither outcome is treated as the primary
result until the placebo, subgroup and contamination controls have been applied
to it on identical terms.

**D2 — 18 August 2026 — source of item text, pre-administration.** §4 anticipated
obtaining ICAR item text from the ICAR project website. That route is closed: the
site's registration path directs to a contact address on a domain that has lapsed,
and the item pages are password-gated. Item text was instead taken from the
official PsychArchives / ZPID release of the ICAR item set. Fidelity is verified
rather than assumed: all eight transcribed answer keys match the published scoring
key shipped in `psychTools`, and the deterministic series items are independently
re-solved from their stems (`scripts/verify_items.py`, a gate that blocks
administration on any mismatch).
