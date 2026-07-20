# Investment engine v3

`matching_v3.py` redesigns the matching methodology so the engine reasons like
an investment analyst. It reuses the v2 infrastructure (embeddings with the
capability-focus blend, the graded self-consistency GPT gate, analyst
overrides, drift-instrumented labels) and replaces the scoring, feature
engineering, ranking and reasoning. Output: `Output/matching_output_v3.csv`.

## Scoring components

Analyst decision (2026-07-20): the GPT-inferred company scores are REPORTED in
the output but carry no weight in final_score. They are inferred from thin
profile text, so weighting them rewards data availability rather than fit.
Only pair evidence scores positively; the inferred layer acts through
subtractive penalties (which can never inflate a score) and stays visible as
columns for the analyst.

| Component | Weight (effective) | Role |
|---|---|---|
| Sector fit | 36.4% | Scores (hierarchical taxonomy similarity) |
| Company profile | 36.4% | Scores (embedding percentile + specificity) |
| Products and services | 27.3% | Scores (embedding percentile + specificity) |
| Value-chain compatibility | 0% | Reported column + subtractive penalties only |
| Investment readiness | 0% | Reported column only |
| Strategic fit | 0% | Reported column only |
| Localization potential | 0% | Reported column only |

Weights are configurable: `--weights my.json` overrides any subset (any zeroed
component can be re-enabled); they are re-normalized to sum to 1.

## Sector taxonomy

Six families (industrial, ict, healthcare, energy, chemicals, mining) with
leaves, a synonym map for raw labels, and cross-family affinities. Similarity:
same leaf 1.0, parent-child 0.8, same-family siblings 0.65, cross-family via
the affinity table (e.g. industrial-ict 0.45, chemicals-healthcare 0.55),
floor 0.10-0.15. Exact-string zero matching is gone: every pair gets a graded
value. Each company also gets a GPT-refined `normalized_sector` leaf from its
profile text (cached), so blanket labels like "Industrial Manufacturing" stop
hiding what a company actually is.

## Value-chain intelligence

Every company is classified (GPT, cached) into one of 12 roles (Raw Material
Supplier, Component Supplier, OEM, Contract Manufacturer, Technology Provider,
Platform Company, Research Company, Distributor, System Integrator, Service
Provider, Investor, Developer) plus an optional secondary role. Every
opportunity declares up to three required roles, ranked. A compatibility
matrix scores the pairing; the spec's calibration cases hold by construction:
drug developer vs research/biotech 0.9, vs contract manufacturer 0.45, vs
chemical supplier 0.15.

## Investment readiness

Fifteen dimensions per company (manufacturing capability, technology
ownership, IP intensity, export capability, global footprint, localization
readiness, regional expansion, GCC presence, greenfield likelihood, JV
potential, partnership potential, R&D capability, capital capacity,
operational maturity, Vision 2030 alignment), GPT-scored 0-1 from the profile
with an explicit conservatism rule: no evidence means 0.3 and lower
classification confidence. Composites: readiness (8 core), strategic fit (4),
localization (4). Cached in `Output/enrichment_cache_v3.json`.

## False-positive penalties

Multiplicative, each recorded on the row: product-only similarity (x0.65),
sector mismatch below 0.25 (x0.75), value-chain mismatch below 0.3 (x0.7),
supplier-to-developer (x0.7), business-model mismatch (x0.7). Penalties stack.

## Decisions and precision

Six tiers: Excellent / Strong / Good / Potential / Weak / Poor Match.
Hierarchy of authority: analyst verdict (Agree floors at Good, Disagree forces
Poor) outranks the AI gate (No caps at Weak, Partial caps at Strong) which
outranks the score. Pairs the gate never examined cap at Potential Match
regardless of score: precision over recall, an unvetted pair is never
presented as a vetted one. `ai_score` is 1 for the top three tiers.

## Confidence

0-100 with a High/Medium/Low label: 30% data completeness, 25% classification
certainty, 25% cross-component agreement, 20% gate vote agreement.

## Output columns (25)

company_id, company_name, opportunity_id, opportunity_name, company_sector,
normalized_sector, opportunity_sector, sector_similarity, profile_similarity,
product_similarity, value_chain_score, investment_readiness_score,
strategic_fit_score, localization_score, ai_score, confidence_score, decision,
final_score, rank, strengths, risks, recommended_engagement,
suggested_localization_model, match_reason, executive_summary.

Narratives are balanced by construction: strengths, risks (including missing
capabilities), recommended engagement, a localization model from a fixed menu,
three factual match reasons, and an executive summary. Stock-phrase vocabulary
is banned in the prompt.

## Performance

Embeddings and their cache are reused from v2. Company and opportunity
classifications are cached by text hash, so repeated runs pay only for the
gate and narratives. Reference run: 61 companies x 12 opportunities in about
five minutes with 8 workers.
