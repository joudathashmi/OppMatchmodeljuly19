# matching_v2.py — design and reference

`matching_v2.py` is the current primary matching pipeline. It scores every
company against every opportunity, ranks candidates in both directions, and
exports a single workbook `Output/matches_v2.xlsx`. It supersedes the older
`business_grade_matching.py` (kept for reference only).

## What it matches

| Side | File | Count | Sectors present |
|------|------|-------|-----------------|
| Companies | `Data/companies.xlsx` | 64 | Industrial Manufacturing (36), General Industry (13), Oil/Gas/Energy/Water (7), Mining (4), Engineering & Construction (3), ICT (1) |
| Opportunities | `Data/new_opportunities.xlsx` | 12 | ICT Hardware (5), Pharmaceutical (4), MedTech (3) |

**The central data reality:** no company sits in Pharmaceutical or MedTech, and
only one in ICT. Seven of twelve opportunities (all Pharma + MedTech) have zero
same-sector companies. Every match for those opportunities therefore depends on
a cross-sector *bridge* (e.g. an industrial precision-manufacturer plausibly
serving MedTech), and where no bridge is real the model is expected to
**abstain** rather than force a top pick. This is the hardest part of the
problem and the source of most open issues.

## Pipeline stages

1. **Load** (`load_companies`, `load_opportunities`) — normalize column names,
   validate required fields, build a preprocessed `combined` text per company
   and a concatenated `requirement` text per opportunity.
2. **Vocabulary fit** (`Vocabulary.fit`) — learn corpus-common tokens to
   suppress, but **protect** sector vocabulary, the sector ontology, and bridge
   capability terms from suppression (this is FIX 1; see below). Also computes
   IDF weights and the domain-keyword set used for evidence.
3. **Vectorize** (`build_vectors`) — OpenAI `text-embedding-3-large` if a
   working key is present, otherwise a hybrid word + character TF-IDF fallback.
   The chosen mode is stamped on every output row (`embed_mode`).
4. **Similarity + calibration** — cosine similarity matrices for profile↔opp and
   product↔opp, then **percentile-rank** normalization so scores are comparable
   across embedding backends (FIX 2). A **specificity** term (how far a pair
   beats that company's own row average) is blended in to fight generalist
   dominance (FIX 8).
5. **Per-pair scoring** — for each of the 768 pairs compute sector score,
   evidence terms/score, the soft-match check, qualification, and the fused
   `final_score`.
6. **Ranking** — `rank_for_opportunity` and `rank_for_company` derived from the
   one scoring table (FIX 3 — single pass, both views).
7. **GPT validation** (optional, `gpt_validate`) — runs once on qualified
   top-N pairs after scores are frozen; the verdict gates the decision label but
   never rewrites the score scale (FIX 9). Verdicts append to
   `Output/gpt_labels.jsonl` as an evaluation set.
8. **Export** — `Output/matches_v2.xlsx` with sheets: Opportunity_View,
   Company_View, All_Pairs, Abstentions, Diagnostics.

## Scoring formula

```
final = 0.30 * semantic_profile   # profile↔opp, percentile+specificity blend
      + 0.30 * semantic_product   # products↔opp, percentile+specificity blend
      + 0.25 * sector_score       # ontology / group / bridge, 0..1
      + 0.15 * evidence_score     # IDF mass of shared capability terms
```

`semantic_* = (1 - SPECIFICITY_BLEND) * global_percentile
            + SPECIFICITY_BLEND * specificity_percentile` with
`SPECIFICITY_BLEND = 0.35`.

### Sector score (`sector_score`)
- Exact text match → 1.0.
- Token overlap (Jaccard) after canonicalization + ontology expansion.
- Same broader industry family (`SECTOR_GROUPS`) → 0.75 floor.
- Otherwise a `BRIDGE_RULES` cross-sector rule may fire (0.45–0.58) if the
  company's capability tokens hit the bridge's term set.
- Labels: Exact / Strong (≥0.80) / Moderate (≥0.50) / Weak (>0) / No.

### Soft-match path
When sector is No/Unknown but there are ≥`MIN_EVIDENCE_TERMS` shared domain
terms **and** semantic percentile ≥ `SOFT_MATCH_MIN_PCT` (0.60), the pair is
promoted to a 0.35 "Weak" sector score. This is what lets genuinely-unfittable
opportunities still surface a candidate — see Known issues.

### Qualification and labels
- `qualified = sector_score >= 0.35 AND n_evidence_terms >= 2`.
- `real_sector_link = label in {Exact, Strong, Moderate} OR bridge fired`.
- `decision_label`: soft matches without a real sector link cap at "Review
  Needed"; GPT "Yes"/"No" overrides to High/Good/Low.
- **Abstention:** an opportunity with no qualified candidate is listed on the
  Abstentions sheet instead of force-ranking the least-bad company (FIX 7).

## Config knobs (top of file)

| Constant | Default | Effect |
|----------|---------|--------|
| `W_PROFILE / W_PRODUCT / W_SECTOR / W_EVIDENCE` | .30/.30/.25/.15 | final-score weights |
| `SPECIFICITY_BLEND` | 0.35 | generalist-dominance correction strength |
| `MIN_SECTOR_SCORE` | 0.35 | qualification floor (equals the soft-match value) |
| `MIN_EVIDENCE_TERMS` | 2 | shared capability terms required to qualify |
| `SOFT_MATCH_MIN_PCT` | 0.60 | semantic percentile needed to soft-pass sector |
| `MIN_EVIDENCE_IDF` | 1.5 | rarity floor for a term to count as evidence |
| `GPT_TOP_N_PER_OPPORTUNITY` | 3 | how many top pairs per opp go to GPT |

## Running

```bash
python3 matching_v2.py                 # auto: OpenAI if key works, else TF-IDF
python3 matching_v2.py --no-gpt        # skip GPT validation
python3 matching_v2.py --no-openai     # force TF-IDF fallback
python3 matching_v2.py --require-openai # fail hard instead of TF-IDF fallback
```

Outputs: `Output/matches_v2.xlsx`, `Output/gpt_labels.jsonl`,
`Output/emb_cache_v2.npz` (embedding cache).

## The nine fixes over v1 (`Code.ipynb`)

1. Sector vocabulary protected from dynamic stopword learning (v1 stripped
   "industrial"/"manufacturing" and silently dropped 36/64 companies).
2. Percentile-normalized semantic scores (backend-independent calibration).
3. Single scoring pass feeds both ranking views (half the compute/GPT spend).
4. Vectorized cosine similarity.
5. Batched + on-disk-cached embeddings.
6. Embedding mode stamped per row; `--require-openai` fails loudly.
7. Abstention instead of forcing a top pick.
8. Specificity/popularity correction so long generic profiles stop winning.
9. GPT gate runs once on frozen scores; verdicts persisted for evaluation.

## Change log / status

**2026-07-15 — soft-match and ranking fixes (landed):**

- **Soft-matches no longer qualify.** A pair with no sector overlap *and* no
  fired bridge (`soft_candidate=True`) is surfaced in All_Pairs for transparency
  but is excluded from `qualified`, so it never reaches the client-facing views.
  This removed ~198 noise pairs and killed indefensible top picks such as a
  power-transformer maker (Maschinenfabrik Reinhausen) topping the MRI and
  Ventilator opportunities. Qualified pairs 465 → 267.
- **Cross-sector bridge matches cap at "Good Fit"** (`bridge_only`) on model
  score alone — only a GPT "Yes" can elevate a bridge-only pair to "High Fit".
- **Ranks are computed over qualified pairs only.** Previously
  `rank_for_opportunity/company` ranked all pairs, so a high-scoring but
  unqualified soft candidate held rank 1 and pushed genuine bridged candidates
  past the view's rank cutoff — making them silently vanish (Ventilator showed
  zero candidates while the Abstentions sheet claimed it had some). Fixed.

Post-fix, all 12 opportunities with a qualified candidate appear in the
Opportunity_View; pharma/medical opps show bridged industrial/chemical
candidates (Chemtrade via Energy/Chemicals↔Pharma, Wuerth/Zhejiang NAMAG via
Industrial↔Medical/Pharma) labeled "Good Fit"/"Review Needed", not forced picks.

## Known issues (open work)

1. **Generalist top-1 (deferred, not a defect in TF-IDF mode).** Belden is #1 on
   all 5 ICT opportunities, but it's the one genuine broad-electronics company
   and the top-3 slates are diverse (6 distinct companies across 15 slots), so
   the dominance is legitimate. Do **not** tune `SPECIFICITY_BLEND` against
   TF-IDF noise; re-evaluate after an OpenAI re-run.
2. **OpenAI path unverified recently.** The key in `.env` has been expired, so
   the model has been running on the TF-IDF fallback. Scores across modes are
   deliberately not comparable; the GPT gate and semantic quality need a re-run
   with a working key (`--require-openai`) to be validated. This is the main
   prerequisite before further scoring tuning.
