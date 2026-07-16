# Code and algorithm audit

Date: 2026-07-16. Scope: `matching_v2.py`, `export_matching_csv.py`,
`build_review_gui.py`. Verdict: the algorithm and layering are sound. The issues
are robustness gaps, one verdict inconsistency, and quality/maintainability debt,
not broken logic.

Status column: implemented items are marked. Fixes landed in the commits
following this document.

## High priority

### H1. No timeout or retries on the OpenAI/Azure clients (FIXED)
`resolve_backends` built `OpenAI()` and `AzureOpenAI(...)` with no timeout, so
one stalled connection could wedge a run for the SDK default of 600 seconds per
request. This is the same failure that hung the exporter for 64 minutes.
Fix: build both clients with a 60s timeout and 3 retries.

### H2. GPT gate ran sequentially (FIXED)
The validation loop called the gate one pair at a time (36 pairs x 3 votes = 108
serial requests). Fix: run the validations through a thread pool; results are
collected in parallel and written to the frame sequentially. Roughly 5x faster.

### H3. Two sources of truth for the match verdict (FIXED)
`matching_v2.py` produced the graded, voted gate verdict; `export_matching_csv.py`
independently re-judged every pair with its own single call and a different
system prompt. They disagreed (18 partner fits vs 118 CSV "Yes"). Fix: the
exporter reuses the gate's persisted verdict for every pair the gate validated,
and uses the same rubric and system prompt for the remaining long-tail pairs, so
the CSV and the workbook never contradict each other on the same pair.

## Medium priority

### M1. final_score mixes relative and absolute components (ADDRESSED)
The semantic terms are percentiles (relative to the run), while sector and
evidence are absolute; they are summed and then compared against fixed thresholds
in `decision_label`. Because the percentile part rescales with the dataset, a
fixed threshold does not mean the same thing on 768 pairs as on 900k.

Addressed by calibration rather than re-thresholding: `calibrate_probability`
fits a logistic mapping from final_score to the gate's accumulated verdicts in
`gpt_labels.jsonl` (latest verdict per pair wins) and adds a calibrated
`match_probability` column to every row. One monotone feature, so the reported
AUC is exactly final_score's rank-discrimination on the label pool - an honest
measure that improves as labels accumulate across runs. First fit: 50 labels,
AUC 0.779. The heuristic label thresholds remain for the no-GPT path but the
probability column is the calibrated signal.

### M2. Evidence terms dominated by generic vocabulary (FIXED)
The top evidence terms were "manufacturing", "industrial", "maintenance",
"engineering", true of almost every company, so they did not discriminate, and
filler words ("used", "units", "local", "centers") leaked through. Cause: these
are protected sector tokens that escape common-word suppression, and the IDF
floor was too soft on a small corpus. Fix: exclude corpus-common terms from
evidence via a document-frequency ceiling, and extend the evidence stoplist.

Important: evidence-for-display is DECOUPLED from evidence-for-qualification. A
first attempt let the stricter filter drive qualification too, which halved the
qualified set and made 5G Small Cell abstain (it lost its legitimate Belden
supplier match). The fix keeps qualification on the coverage-preserving lenient
count, and applies the strict discriminating filter only to the shown evidence
and the evidence score. Verified: 18 partner fits and 3 abstentions, identical to
the pre-audit baseline, with clean evidence terms on every partner fit
(e.g. Belden: "cabling, fiber, enclosures, cables"; Chemtrade: "chemicals,
specialty, chemical").

### M3. O(companies x opportunities) redundant tokenization (FIXED)
`domain_overlap` re-tokenized the company text and the opportunity text on every
pair (each company 12x, each opportunity 64x). Negligible at 64x12, a bottleneck
at the real 2,960 x 309 scale (~900k pairs). Fix: precompute each company's and
each opportunity's token set once before the pair loop.

### M4. Dead soft-match mutation (FIXED)
The soft-match block set `soft_match=True` and rewrote the sector score and label,
but qualification excludes soft matches, so the block only relabelled rows that
were then hidden. Fix: keep the `soft_candidate` flag for transparency, drop the
misleading state mutation.

### M5. No tests (FIXED)
Added `tests/test_matching.py` covering `sector_score`, the bridge rules,
`decision_label`, `percentile_rank`, the focus-text stripping, and the gate vote
aggregation. Runs with plain `python3 tests/test_matching.py` (no pytest needed).

## Low priority

### L1. main() is a ~270-line monolith (not yet applied)
Does load, embed, score, gate, abstain, and export in one function. Decomposing
into `score_pairs()` / `run_gate()` / `write_output()` would improve testability.
Deferred: it is a pure refactor with regression risk and no behaviour change.

### L2. Stale docstrings and comments (FIXED)
The top-of-file FIX list and the BRIDGE_RULES comment predated the graded gate,
self-consistency voting, focus blend, and Azure support. Refreshed.

### L3. Embedding cache reloaded on every call (FIXED)
`embed_texts` reloaded the entire on-disk cache on each of its five calls per run.
Fix: load the cache once per run and save once.

### L4. View shows top-5 but validates top-3 (documented)
The opportunity view shows the top-5 qualified candidates, but only the top-3 are
GPT-validated; ranks 4-5 are capped to "Review Needed". Consistent and safe, but
if a fully validated view is wanted, raise the validation depth to match the view.

## Calibration note
In the gate, a tied vote resolves to the more conservative tier. This is
deliberately conservative; noted so it is a choice, not an accident.

## World-class pass 2 (2026-07-16, post-audit)

Three further upgrades beyond the audit items:

1. **Entity resolution.** Duplicate company rows (same entity entered twice,
   e.g. "Tuwaiq Casting & forging" / "Tuwaiq Casting and Forging") split scores
   and occupied two rank slots. `canonical_name` (casefold, &->and, punctuation
   and trailing legal suffixes stripped) merges them at load, keeping the row
   with the richest text and reporting every merge. 3 rows merged; 64 -> 61
   companies. `--no-dedupe` restores the old behaviour.
2. **Vote escalation.** A split first round now draws 2 extra samples before
   tallying, so borderline pairs are decided on 5 votes instead of 3. On the
   verification run, 33/34 pairs were unanimous and the single split pair
   settled at 4/5; gate_unanimous_share is reported in Diagnostics (0.971).
3. **Calibrated match_probability** (see M1 above), with calibration_labels and
   calibration_auc_final_score reported per run.
