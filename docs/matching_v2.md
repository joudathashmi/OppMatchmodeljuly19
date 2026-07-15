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
- **Graded GPT gate.** For each qualified top-N pair the gate returns a tier via
  self-consistency voting (`GPT_VOTES` samples, majority wins, ties resolve to
  the more conservative tier). The `gpt_agreement` column ("k/n") and a
  confidence = winning-share × mean-stated-confidence make the verdict calibrated
  and stable across runs.
  - **Direct** — the company could itself manufacture/assemble/deliver the
    opportunity's product → **High Fit** (final ≥ 0.60) / **Good Fit**.
  - **Partial** — a credible supplier/partner/adjacent player with a named
    component, material, or technology linkage → **Partner Fit**.
  - **None** — only generic sector/keyword overlap → **Low Fit**.
- `decision_label`: without a GPT verdict the score ladder applies (soft/no-link
  → Review Needed; bridge-only caps at Good Fit). When GPT ran, a GPT-unseen
  qualified pair cannot keep a model-only Good/High Fit and is capped at Review
  Needed, so no sheet claims a fit GPT has not confirmed.
- **Abstention:** an opportunity is listed on the Abstentions sheet when it has
  no qualified candidate, or (GPT mode) when no validated candidate is graded
  Direct or Partial — instead of force-ranking the least-bad company.

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
| `GPT_VOTES` | 3 | self-consistency samples per pair (`--gpt-votes`) |
| `GPT_TEMPERATURE` | 0.3 | sampling temperature for the ensemble |
| `GPT_MODELS` | gpt-4.1 first | public-API gate models, best-first |

## Running

```bash
python3 matching_v2.py                 # auto: OpenAI if key works, else TF-IDF
python3 matching_v2.py --no-gpt        # skip GPT validation
python3 matching_v2.py --no-openai     # force TF-IDF fallback
python3 matching_v2.py --require-openai # fail hard instead of TF-IDF fallback
python3 matching_v2.py --env-file PATH  # load extra .env (e.g. shared Azure creds)
python3 matching_v2.py --chat-provider public  # run the gate on public gpt-4.1
python3 matching_v2.py --gpt-votes 5     # more self-consistency samples
```

**Recommended world-class run** (real embeddings + gpt-4.1 gate + voting):

```bash
python3 matching_v2.py --env-file "/Users/joudathashmi/Downloads/uhnwi-fastapi 1/.env" \
  --chat-provider public
```

Outputs: `Output/matches_v2.xlsx`, `Output/gpt_labels.jsonl`,
`Output/emb_cache_v2.npz` (embedding cache).

### LLM backends (public OpenAI or Azure OpenAI)

`resolve_backends()` picks the chat and embeddings backends **independently**,
so GPT validation can run on Azure while semantic vectors come from the public
API (or TF-IDF). Priority: Azure (when `MISA_USE_AZURE_OPENAI=true` and endpoint
+ key are set) for whichever of chat/embeddings has a configured deployment;
the public `OPENAI_API_KEY` fills the rest; TF-IDF is the final fallback for
vectors. On Azure the `model=` argument is a **deployment name**, not a model
family.

Environment variables (same convention as the uhnwi-fastapi project):

| Var | Purpose |
|-----|---------|
| `MISA_USE_AZURE_OPENAI` | `true` to enable the Azure path |
| `AZURE_OPENAI_ENDPOINT` | e.g. `https://<resource>.openai.azure.com` |
| `AZURE_OPENAI_API_KEY` | Azure key |
| `AZURE_OPENAI_API_VERSION` | defaults to `2024-08-01-preview` |
| `AZURE_OPENAI_DEPLOYMENT` | **chat** deployment (e.g. `gpt-4.1-mini`) |
| `AZURE_OPENAI_EMBED_DEPLOYMENT` | **embeddings** deployment (optional) |
| `OPENAI_API_KEY` | public-API key (used for whatever Azure doesn't cover) |

Reuse the uhnwi Azure credentials without copying secrets:

```bash
python3 matching_v2.py --env-file "/Users/joudathashmi/Downloads/uhnwi-fastapi 1/.env"
```

Note: the `merketfit.openai.azure.com` resource has a chat deployment
(`gpt-4.1-mini`) but **no embeddings deployment**, so a run reusing those creds
gets GPT on Azure and embeddings from the public `OPENAI_API_KEY` (or TF-IDF if
that key is dead). Add a `text-embedding-3-large` deployment to the Azure
resource if you need embeddings inside the Azure tenant too.

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

**2026-07-15 — graded self-consistency gate (world-class pass, landed):**

- **Self-consistency voting.** LLM verdicts are non-deterministic even at
  temperature 0 (an A/B test caught `gpt-4.1-mini` flipping the same pair Yes↔No
  across runs). The gate now samples `GPT_VOTES` (3) times and majority-votes,
  reporting `gpt_agreement` and a calibrated confidence. On the reference run
  agreement is 3/3 on almost every pair — the verdict is now stable.
- **gpt-4.1 gate.** The A/B showed `gpt-4.1` is the most precise/consistent
  tier; `GPT_MODELS` lists it first and `--chat-provider public` routes the gate
  to it (Azure `gpt-4.1-mini` remains available for in-tenant residency).
- **Graded 3-tier verdict (Direct / Partial / None)** with a rubric prompt,
  replacing binary Yes/No. This resolved a real failure: under a strict binary
  "can it execute?" gate every opportunity abstained (0 fits) because no company
  in the roster is a finished-product assembler. Grading recognizes credible
  **supplier/partner** fits. Reference run: 0 Direct, **17 Partner fits across 9
  opportunities** (e.g. Chemtrade supplies API-synthesis reagents; Belden
  supplies 5G cabling/connectivity), **3 true abstentions** (MRI, Ventilator,
  Regional refurbishment — no credible partner). New label: **Partner Fit**.
- **Label integrity:** Good/High Fit require a GPT verdict; GPT-unseen qualified
  pairs are capped at Review Needed.

**2026-07-15 — Azure backend + GPT-aware triage (landed):**

- **Azure OpenAI support** — see the "LLM backends" section above.
- **First full real run** (real `text-embedding-3-large` embeddings + Azure
  `gpt-4.1-mini` gate): GPT rejected 35 of 36 qualified top-3 pairs, accepting
  exactly one (Belden → 5G Small Cell, conf 0.90). The rejections are
  well-reasoned (e.g. Chemtrade = inorganic sulfur chemicals ≠ pharma-API
  organic synthesis). Takeaway: the embedding+bridge layer is decent **recall**
  but weak **precision**; GPT is the real precision layer. This company roster
  (industrial/energy/mining) genuinely has ~one strong fit for the ICT/pharma/
  medtech opportunity set — the honest output is triage, not a forced top-3.
- **GPT-aware abstention** — an opportunity now abstains not only when it has no
  qualified candidate, but also when GPT rejected *every* validated top-N
  candidate ("No validated fit (all GPT-rejected)"). On the real run this flags
  11 of 12 opportunities, naming the best (rejected) candidate for each.
- **Bridge tightening** — every bridge now needs `>= min_hits` (2) distinct
  capability terms, and at least one from *outside* `GENERIC_BRIDGE_TERMS`, so a
  couple of boilerplate words ("precision", "chemical") no longer bridge an
  industrial company into pharma/medtech. Qualified pairs 267 → 199.

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

1. **Supply-side coverage is the real limiter, not the model.** The company
   roster is Industrial/Energy/Mining; the opportunities are ICT/Pharma/MedTech.
   On the real GPT-gated run, 11 of 12 opportunities have no validated fit. No
   scoring change fixes a missing supply side — if pharma/medtech matches are
   wanted, add pharma/medtech-capable companies to `Data/companies.xlsx`.
2. **Azure resource has no embeddings deployment.** `merketfit.openai.azure.com`
   serves chat only; embeddings currently come from the public `OPENAI_API_KEY`.
   Add a `text-embedding-3-large` deployment if embeddings must stay in-tenant.
3. **GPT model tier.** Validation runs on `gpt-4.1-mini`. Its reasoning looks
   sound (it accepts the one genuine fit and rejects the rest with specific
   rationale), but a stronger model (`gpt-4o`/`gpt-4.1`) could be A/B-tested on
   the borderline ICT bridges (Eaton, Riyadh Cables) to confirm none are being
   rejected too harshly.

*Resolved:* the earlier "Belden generalist dominance" concern is moot — on the
real embedding run the ICT slates are diverse and GPT-aware abstention plus the
GPT gate now govern what surfaces as a validated fit.
