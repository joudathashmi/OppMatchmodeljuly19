# AI-Powered Company x Opportunity Matching Framework

Scalable, Explainable, Sector-Aware, and Self-Validating.

This is the current eight-step pipeline as implemented in `matching_v2.py`. It
updates the earlier framework slide: steps 1, 2, and 8 are unchanged, step 6 is
upgraded, and steps 3, 4, 5, and 7 have changed.

## Step 1: Preprocessing

- Cleans company and opportunity text: lowercasing, punctuation and noise
  removal, whitespace normalization.
- Standardizes unstructured content for embedding-based comparison.
- Strips corporate boilerplate (founding year, headquarters, ticker) and
  opportunity price and geography noise, so capability meaning dominates the
  embedding.

## Step 2: Sector Ontology Expansion

- Expands primary sector tags using a domain-specific sector tree and token
  canonicalization.
- Ensures broader coverage across related sub-domains.
- Example: "information and communication" expands to ICT, telecom, network,
  hardware, semiconductor.

## Step 3: Sector Scoring and Cross-Sector Bridges

*(Changed from "Sector Filtering".)*

- Does not hard-filter. Every pair is scored, so a valid cross-sector match is
  never dropped before it can be judged.
- Sector score comes from an exact match, an industry-family overlap, or a
  structured cross-sector bridge (for example Industrial to ICT, or Energy and
  Chemicals to Pharma).
- A bridge requires at least two distinct, non-generic capability terms, which
  blocks spurious links (a single word like "chemical" is not enough).

## Step 4: Semantic Embedding and Similarity

*(Changed: model and calibration.)*

- Uses `text-embedding-3-large` to convert text into 3,072-dimension vectors
  (the earlier framework used `ada-002` at 1,536 dimensions).
- Captures the contextual meaning of company profiles and opportunity
  descriptions.
- Cosine similarity, then percentile normalization so scores are comparable
  across runs and embedding backends.
- A capability-focused blend sharpens the signal without changing ranked
  outcomes.

## Step 5: Product and Service Matching

*(Changed: fusion method.)*

- Separately embeds the products and services text for an added, product-level
  signal.
- Combined score is a weighted blend (profile 30 percent, product 30 percent,
  sector 25 percent, evidence 15 percent), not a simple maximum of profile and
  product.
- Goal: detect deeper product-level alignment, not only high-level profile
  overlap.

## Step 6: GPT-Based Validation, Graded and Voted

*(Upgraded.)*

- Asks: can this company realistically deliver or supply this opportunity?
- Returns a graded verdict, not a plain yes or no: Direct (can execute itself),
  Partial (a credible supplier or partner), or None.
- Self-consistency: the model is sampled three times and the majority verdict
  wins, with the agreement ratio reported as a calibrated confidence.
- Goal: human-grade evaluation and a written explanation for every surfaced
  match.

## Step 7: Abstention and Evidence Guardrails

*(Replaces "Soft Match Mode".)*

- Pure semantic similarity with no sector link no longer forces a match. That
  path produced false positives (for example a transformer maker topping an MRI
  opportunity), so it is disqualified.
- Cross-sector value is instead captured through the structured bridges in
  Step 3 and the gate in Step 6.
- If no candidate is validated as Direct or Partial, the opportunity abstains
  and reports "no validated fit" rather than forcing a weak pick.

## Step 8: Ranking and Scoring

- Ranks matches by the blended final score (top rank equals 1).
- Good Fit and High Fit require a GPT verdict, so labels never overstate.
- Goal: a clean, honest shortlist for business teams, with a graded verdict, a
  confidence value, and a written rationale for every match.
