#!/usr/bin/env python3
"""Investment-grade matching engine (v3).

Redesign of the matching methodology so the engine reasons like an investment
analyst, not a semantic search engine. Reuses the proven v2 infrastructure
(embeddings + focus blend, graded GPT gate with self-consistency voting,
analyst overrides, drift-instrumented labels) and adds:

  1. Hierarchical sector taxonomy: synonym normalization, parent-child
     relationships, cross-family affinities. Sector similarity is graded, never
     an exact-string 0/1.
  2. Value-chain intelligence: every company is classified into a value-chain
     role (GPT, cached); every opportunity declares the roles it needs; a
     compatibility matrix scores the pairing (a raw-material supplier is not a
     drug developer).
  3. Investment readiness: 15 sub-dimensions scored per company from its
     profile (GPT, cached, conservative on missing evidence), composed into
     readiness / strategic-fit / localization scores.
  4. Configurable weighted scoring (defaults: sector 20, profile 20, product
     15, value chain 15, readiness 15, strategic 10, localization 5) with
     explicit false-positive penalties, each recorded on the row.
  5. Six-tier decisions (Excellent/Strong/Good/Potential/Weak/Poor Match),
     confidence 0-100, and balanced narratives: strengths, risks, recommended
     engagement, suggested localization model, executive summary.

Usage:
  python3 matching_v3.py                     # full run -> Output/matching_output_v3.csv
  python3 matching_v3.py --no-narratives     # scores + decisions only (fast)
  python3 matching_v3.py --weights my.json   # override scoring weights
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from matching_v2 import (
    GPT_MODELS, LABELS_JSONL, RUBRIC_HASH, SPECIFICITY_BLEND,
    build_exemplar_lines, build_vectors, gpt_validate, load_companies,
    load_human_reviews, load_opportunities, load_prior_verdicts, percentile_rank,
    resolve_backends,
)
from datetime import datetime, timezone

OUTPUT_CSV = "Output/matching_output_v3.csv"
ENRICH_CACHE = "Output/enrichment_cache_v3.json"

# ------------------------------ sector taxonomy ------------------------------

# family -> leaves. A label may also normalize to the family itself.
TAXONOMY = {
    "industrial": {"industrial manufacturing", "electrical equipment",
                   "industrial automation", "robotics", "machinery",
                   "metals & fabrication", "engineering & construction",
                   "industrial equipment"},
    "ict": {"ict hardware", "electronics", "telecom equipment", "software",
            "semiconductors"},
    "healthcare": {"pharmaceutical manufacturing", "biotechnology",
                   "medical devices", "healthcare services"},
    "energy": {"oil & gas", "power generation", "renewables", "utilities",
               "water"},
    "chemicals": {"industrial chemicals", "specialty chemicals",
                  "petrochemicals"},
    "mining": {"mining & minerals", "metals"},
}
LEAF_TO_FAMILY = {leaf: fam for fam, leaves in TAXONOMY.items() for leaf in leaves}
ALL_LEAVES = sorted(LEAF_TO_FAMILY) + sorted(TAXONOMY)

SECTOR_SYNONYMS = {
    "industrial manufacturing": "industrial manufacturing",
    "general industry": "industrial",
    "engineering & construction": "engineering & construction",
    "ict": "ict",
    "ict hardware": "ict hardware",
    "information and communication technology": "ict",
    "electrical equipment": "electrical equipment",
    "medtech": "medical devices",
    "medical devices": "medical devices",
    "pharmaceutical": "pharmaceutical manufacturing",
    "pharmaceutical manufacturing": "pharmaceutical manufacturing",
    "pharma": "pharmaceutical manufacturing",
    "biotechnology": "biotechnology",
    "healthcare": "healthcare",
    "healthcare and life sciences": "healthcare",
    "oil, gas, energy & water": "energy",
    "oil & gas": "oil & gas",
    "energy": "energy",
    "mining": "mining & minerals",
    "chemicals": "industrial chemicals",
}

# symmetric cross-family investment affinity (industrial firms plausibly extend
# into ICT hardware assembly; chemicals feed pharma; etc.)
FAMILY_AFFINITY = {
    frozenset(("industrial", "ict")): 0.45,
    frozenset(("industrial", "healthcare")): 0.35,
    frozenset(("industrial", "energy")): 0.45,
    frozenset(("industrial", "chemicals")): 0.40,
    frozenset(("industrial", "mining")): 0.40,
    frozenset(("chemicals", "healthcare")): 0.55,
    frozenset(("energy", "chemicals")): 0.50,
    frozenset(("ict", "healthcare")): 0.40,
    frozenset(("mining", "chemicals")): 0.40,
    frozenset(("mining", "energy")): 0.45,
    frozenset(("ict", "energy")): 0.30,
    frozenset(("energy", "healthcare")): 0.20,
    frozenset(("ict", "chemicals")): 0.20,
    frozenset(("mining", "ict")): 0.15,
    frozenset(("mining", "healthcare")): 0.15,
    frozenset(("mining", "industrial")): 0.40,
}
BASE_CROSS_FAMILY = 0.10


def normalize_sector_label(raw: str) -> str:
    """Map a raw sector string to a taxonomy node via synonyms (fallback path;
    the enrichment step refines this from the full profile text)."""
    key = re.sub(r"\s+", " ", str(raw or "").strip().lower())
    if key in SECTOR_SYNONYMS:
        return SECTOR_SYNONYMS[key]
    if key in LEAF_TO_FAMILY or key in TAXONOMY:
        return key
    return ""


def _family(node: str) -> str:
    return node if node in TAXONOMY else LEAF_TO_FAMILY.get(node, "")


def sector_similarity(node_a: str, node_b: str) -> float:
    """Graded taxonomy similarity: leaf identity 1.0, parent-child 0.8,
    same-family siblings 0.65, cross-family via affinity table."""
    a, b = node_a or "", node_b or ""
    if not a or not b:
        return 0.30  # unknown: neutral, not zero
    if a == b:
        return 1.0
    fa, fb = _family(a), _family(b)
    if fa and fa == fb:
        if a == fa or b == fb:  # one is the family root
            return 0.80
        return 0.65
    if fa and fb:
        return FAMILY_AFFINITY.get(frozenset((fa, fb)), BASE_CROSS_FAMILY)
    return 0.15

# --------------------------- value-chain intelligence --------------------------

ROLES = ["Raw Material Supplier", "Component Supplier", "OEM",
         "Contract Manufacturer", "Technology Provider", "Platform Company",
         "Research Company", "Distributor", "System Integrator",
         "Service Provider", "Investor", "Developer"]

# required role -> {company role: compatibility}. Missing entries default 0.2;
# identical roles default 1.0.
ROLE_COMPAT = {
    "Contract Manufacturer": {"OEM": 0.85, "System Integrator": 0.60,
                              "Component Supplier": 0.50, "Technology Provider": 0.50,
                              "Developer": 0.40, "Raw Material Supplier": 0.30,
                              "Research Company": 0.30, "Service Provider": 0.30,
                              "Platform Company": 0.25, "Distributor": 0.20,
                              "Investor": 0.15},
    "OEM": {"Contract Manufacturer": 0.85, "Component Supplier": 0.55,
            "Technology Provider": 0.55, "System Integrator": 0.50,
            "Developer": 0.40, "Raw Material Supplier": 0.25,
            "Research Company": 0.30, "Distributor": 0.20, "Investor": 0.15},
    "Component Supplier": {"OEM": 0.70, "Contract Manufacturer": 0.70,
                           "Raw Material Supplier": 0.45, "Technology Provider": 0.50,
                           "System Integrator": 0.40},
    "Developer": {"Research Company": 0.90, "Platform Company": 0.60,
                  "Technology Provider": 0.60, "Contract Manufacturer": 0.45,
                  "OEM": 0.40, "Investor": 0.35, "Component Supplier": 0.25,
                  "Service Provider": 0.20, "Raw Material Supplier": 0.15,
                  "Distributor": 0.15},
    "Research Company": {"Developer": 0.85, "Technology Provider": 0.60,
                         "Platform Company": 0.50, "Contract Manufacturer": 0.35},
    "Technology Provider": {"Platform Company": 0.70, "Research Company": 0.60,
                            "System Integrator": 0.55, "OEM": 0.50,
                            "Developer": 0.50},
    "System Integrator": {"Technology Provider": 0.55, "OEM": 0.60,
                          "Service Provider": 0.50, "Contract Manufacturer": 0.55},
    "Raw Material Supplier": {"Component Supplier": 0.5},
}


def value_chain_score(required_roles: list, company_role: str,
                      secondary_role: str = "") -> float:
    """Best compatibility of the company's role(s) against the opportunity's
    required roles (ranked; later roles slightly discounted)."""
    if not required_roles or not company_role:
        return 0.30
    best = 0.0
    for i, need in enumerate(required_roles[:3]):
        discount = 1.0 - 0.1 * i
        for role in filter(None, [company_role, secondary_role]):
            if role == need:
                score = 1.0
            else:
                score = ROLE_COMPAT.get(need, {}).get(role, 0.20)
            best = max(best, score * discount)
    return round(best, 3)

# --------------------------- investment readiness ---------------------------

READINESS_DIMS = [
    "manufacturing_capability", "technology_ownership", "ip_intensity",
    "export_capability", "global_footprint", "localization_readiness",
    "regional_expansion", "gcc_presence", "greenfield_likelihood",
    "jv_potential", "partnership_potential", "rd_capability",
    "capital_capacity", "operational_maturity", "vision2030_alignment",
]
CORE_DIMS = ["manufacturing_capability", "technology_ownership", "ip_intensity",
             "export_capability", "global_footprint", "rd_capability",
             "capital_capacity", "operational_maturity"]
STRATEGIC_DIMS = ["jv_potential", "partnership_potential", "regional_expansion",
                  "vision2030_alignment"]
LOCALIZATION_DIMS = ["localization_readiness", "greenfield_likelihood",
                     "gcc_presence", "export_capability"]


def compose_readiness(dims: dict):
    def mean_of(keys):
        vals = [float(dims.get(k, 0.3)) for k in keys]
        return round(sum(vals) / len(vals), 3)
    return mean_of(CORE_DIMS), mean_of(STRATEGIC_DIMS), mean_of(LOCALIZATION_DIMS)

# ------------------------------- scoring model -------------------------------

DEFAULT_WEIGHTS = {
    "sector": 0.20, "profile": 0.20, "product": 0.15, "value_chain": 0.15,
    "readiness": 0.15, "strategic": 0.10, "localization": 0.05,
}

# false-positive penalties: (name, predicate(row) -> bool, factor)
def compute_penalties(sector_sim, profile_sem, product_sem, vc_score,
                      company_role, required_roles) -> tuple:
    penalties = []
    if product_sem >= 0.70 and profile_sem < 0.45 and sector_sim < 0.40:
        penalties.append(("product_only_similarity", 0.65))
    if sector_sim < 0.25:
        penalties.append(("sector_mismatch", 0.75))
    if vc_score < 0.30:
        penalties.append(("value_chain_mismatch", 0.70))
    lead_need = (required_roles or [""])[0]
    if (company_role in ("Raw Material Supplier", "Distributor")
            and lead_need in ("Developer", "Research Company", "OEM")):
        penalties.append(("supplier_to_developer", 0.70))
    if (company_role in ("Investor", "Service Provider")
            and lead_need in ("Contract Manufacturer", "OEM")):
        penalties.append(("business_model_mismatch", 0.70))
    factor = 1.0
    for _, f in penalties:
        factor *= f
    return factor, [name for name, _ in penalties]


TIERS = [(0.72, "Excellent Match"), (0.60, "Strong Match"), (0.50, "Good Match"),
         (0.38, "Potential Match"), (0.26, "Weak Match"), (0.0, "Poor Match")]
TIER_ORDER = [t for _, t in TIERS]


def tier_for(score: float) -> str:
    for cut, name in TIERS:
        if score >= cut:
            return name
    return "Poor Match"


def cap_tier(tier: str, cap: str) -> str:
    return tier if TIER_ORDER.index(tier) >= TIER_ORDER.index(cap) else cap


def decide(final: float, gate: str, human: str, gated: bool) -> str:
    """Six-tier decision. Analyst verdicts outrank the gate; the gate outranks
    the score; ungated pairs cannot claim the top two tiers."""
    if human == "Disagree":
        return "Poor Match"
    tier = tier_for(final)
    if human == "Agree":
        # analyst approval sets a FLOOR of Good Match
        return tier if TIER_ORDER.index(tier) <= TIER_ORDER.index("Good Match") else "Good Match"
    if gate == "No":
        return "Weak Match" if TIER_ORDER.index(tier) < TIER_ORDER.index("Weak Match") else tier
    if gate == "Partial":
        return "Strong Match" if tier == "Excellent Match" else tier
    if gate in ("Direct", "Yes"):
        return tier
    # Ungated pairs were never examined by the AI gate: precision over recall
    # means an unvetted pair is at most a "Potential Match", whatever its score.
    return "Potential Match" if TIER_ORDER.index(tier) < TIER_ORDER.index("Potential Match") else tier


def confidence_score(comp_len, opp_len, class_conf, components, gate_agreement) -> int:
    """0-100. Data completeness, classification certainty, cross-component
    agreement, and gate vote agreement."""
    completeness = min(1.0, comp_len / 600) * 0.5 + min(1.0, opp_len / 1500) * 0.5
    vals = np.array(components, dtype=float)
    agreement = 1.0 - min(1.0, float(vals.std()) * 2.0)
    if "/" in str(gate_agreement):
        k, n = gate_agreement.split("/")
        vote = int(k) / max(1, int(n))
    else:
        vote = 0.5
    score = 0.30 * completeness + 0.25 * float(class_conf) + 0.25 * agreement + 0.20 * vote
    return int(round(100 * min(1.0, max(0.0, score))))


def confidence_label(c: int) -> str:
    return "High" if c >= 70 else "Medium" if c >= 45 else "Low"

# ------------------------------- GPT enrichment -------------------------------

_cache_lock = threading.Lock()


def _load_cache() -> dict:
    try:
        with open(ENRICH_CACHE) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_cache(cache: dict):
    os.makedirs(os.path.dirname(ENRICH_CACHE), exist_ok=True)
    with open(ENRICH_CACHE, "w") as fh:
        json.dump(cache, fh)


def _chat_json(client, models, system, prompt):
    for model in models:
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.2,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}])
            content = re.sub(r"^```(?:json)?|```$", "",
                             (resp.choices[0].message.content or "").strip()).strip()
            return json.loads(content)
        except Exception:
            continue
    return None


COMPANY_ENRICH_SYSTEM = (
    "You are an investment analyst classifying companies for FDI targeting. "
    "Judge ONLY from the text. Where the text gives no evidence for a dimension, "
    "score it 0.3 and lower classification_confidence; never invent facts.")


def company_enrich_prompt(comp) -> str:
    leaves = ", ".join(sorted(set(ALL_LEAVES)))
    roles = ", ".join(ROLES)
    dims = ", ".join(READINESS_DIMS)
    return f"""Classify this company. Return STRICT JSON only:
{{"normalized_sector": "<one of: {leaves}>",
  "value_chain_role": "<one of: {roles}>",
  "secondary_role": "<one of the roles or empty string>",
  "size_class": "SME|Mid|Large|Enterprise",
  "classification_confidence": 0.0-1.0,
  "readiness": {{<each of: {dims}> : 0.0-1.0}}}}

COMPANY
  Name: {comp['company_name']}
  Sector label: {comp['Sector']}
  Profile: {comp['company_profile']}
  Products/Services: {comp['product and Services']}
"""


OPP_ENRICH_SYSTEM = (
    "You are an investment analyst decomposing an investment opportunity. "
    "Judge only from the text; return strict JSON.")


def opp_enrich_prompt(opp) -> str:
    leaves = ", ".join(sorted(set(ALL_LEAVES)))
    roles = ", ".join(ROLES)
    return f"""What does this opportunity need? Return STRICT JSON only:
{{"normalized_sector": "<one of: {leaves}>",
  "required_roles": ["<up to 3 of: {roles}, ranked most-needed first>"],
  "stage_needed": "<short phrase for the value-chain stage sought>"}}

OPPORTUNITY
  Name: {opp['What is the opportunity name?']}
  Sector label: {opp['Sector']}
  Description: {opp['What is the opportunity description?']}
  Required materials: {opp['What materials are involved or required in the project?']}
  Value proposition: {opp['What is the value proposition of this opportunity?']}
"""


def enrich_all(client, models, companies, opps, workers=8):
    cache = _load_cache()

    def key_for(text):
        return hashlib.md5(("enrichv1::" + text).encode()).hexdigest()

    def run_company(comp):
        k = key_for(str(comp["company_profile"]) + str(comp["product and Services"]))
        with _cache_lock:
            if k in cache:
                return comp["company_name"], cache[k]
        out = _chat_json(client, models, COMPANY_ENRICH_SYSTEM,
                         company_enrich_prompt(comp))
        if out is None:
            out = {"normalized_sector": normalize_sector_label(comp["Sector"]),
                   "value_chain_role": "", "secondary_role": "", "size_class": "",
                   "classification_confidence": 0.2, "readiness": {}}
        with _cache_lock:
            cache[k] = out
        return comp["company_name"], out

    def run_opp(opp):
        k = key_for(str(opp["What is the opportunity description?"])
                    + str(opp["What is the opportunity name?"]))
        with _cache_lock:
            if k in cache:
                return opp["What is the opportunity name?"], cache[k]
        out = _chat_json(client, models, OPP_ENRICH_SYSTEM, opp_enrich_prompt(opp))
        if out is None:
            out = {"normalized_sector": normalize_sector_label(opp["Sector"]),
                   "required_roles": ["Contract Manufacturer"], "stage_needed": ""}
        with _cache_lock:
            cache[k] = out
        return opp["What is the opportunity name?"], out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        comp_enrich = dict(ex.map(run_company, [c for _, c in companies.iterrows()]))
        opp_enrich = dict(ex.map(run_opp, [o for _, o in opps.iterrows()]))
    _save_cache(cache)
    return comp_enrich, opp_enrich

# ------------------------------- narratives ---------------------------------

NARRATIVE_SYSTEM = (
    "You are a senior investment advisor at MISA writing balanced, decision-"
    "ready assessments. Ground every claim in the given texts; name products, "
    "stages and gaps. Formulaic phrasing is a defect. BANNED: 'strong partner', "
    "'reliable supplier', 'aligns well', 'well-positioned', 'leveraging', "
    "'expertise in', 'proven track record'.")


def narrative_prompt(comp, opp, decision, vc_role, required_roles) -> str:
    return f"""Write the investment assessment for this pairing. The decision is
already made: "{decision}". Company value-chain role: {vc_role or 'unknown'};
the opportunity needs: {', '.join(required_roles or ['unknown'])}.

STYLE RULES (violating any is a defect):
- NEVER use these words/phrases: "expertise in", "leveraging", "leverage",
  "strong partner", "reliable supplier", "aligns well", "well-positioned",
  "proven track record", "extensive experience". Say what the company DOES
  and MAKES instead of characterizing its expertise.
- Ground every claim in the texts: name products, value-chain stages,
  materials, and market facts. A sentence that could be pasted into another
  company's assessment unchanged must be rewritten.

Return STRICT JSON only:
{{"strengths": "2-3 sentences: what genuinely matches, citing named products and value-chain stages",
  "risks": "2-3 sentences: concerns, missing capabilities, incompatibilities; be specific",
  "recommended_engagement": "1-2 sentences: how Invest Saudi should engage (or why not to)",
  "suggested_localization_model": "one of: Greenfield manufacturing | Regional assembly | Joint venture | Licensing and technology transfer | Supplier localization | Distribution partnership | Not recommended",
  "match_reason": ["3 factual reasons for the decision", "...", "..."],
  "executive_summary": "2-3 sentences an investment manager reads first; balanced, specific, decisive"}}

COMPANY
  Name: {comp['company_name']}
  Sector: {comp['Sector']}
  Profile: {comp['company_profile']}
  Products/Services: {comp['product and Services']}

OPPORTUNITY
  Name: {opp['What is the opportunity name?']}
  Sector: {opp['Sector']}
  Description: {opp['What is the opportunity description?']}
  Highlights: {opp['What are the investment highlights?']}
  Required materials: {opp['What materials are involved or required in the project?']}
"""


LOCALIZATION_MENU = ["Greenfield manufacturing", "Regional assembly",
                     "Joint venture", "Licensing and technology transfer",
                     "Supplier localization", "Distribution partnership",
                     "Not recommended"]


def normalize_localization(value: str, decision: str) -> str:
    """Coerce a free-text localization answer onto the fixed menu. The model
    occasionally answers with something else (e.g. a value-chain role); an
    off-menu value falls back deterministically by decision tier."""
    v = str(value or "").strip().lower()
    for item in LOCALIZATION_MENU:
        if item.lower() == v or item.lower() in v or (v and v in item.lower()):
            return item
    if "not recommend" in v:
        return "Not recommended"
    positive = decision in ("Excellent Match", "Strong Match", "Good Match")
    return "Supplier localization" if positive else "Not recommended"


def generate_narrative(client, models, comp, opp, decision, vc_role, required_roles):
    out = _chat_json(client, models, NARRATIVE_SYSTEM,
                     narrative_prompt(comp, opp, decision, vc_role, required_roles))
    if not isinstance(out, dict):
        out = {}
    reasons = out.get("match_reason", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    reasons = [str(r).strip() for r in reasons if str(r).strip()][:3]
    return {
        "strengths": str(out.get("strengths", "")).strip(),
        "risks": str(out.get("risks", "")).strip(),
        "recommended_engagement": str(out.get("recommended_engagement", "")).strip(),
        "suggested_localization_model": normalize_localization(
            out.get("suggested_localization_model", ""), decision),
        "match_reason": json.dumps(reasons, ensure_ascii=False) if reasons else "",
        "executive_summary": str(out.get("executive_summary", "")).strip(),
    }

# ----------------------------------- main -----------------------------------

COLUMNS = [
    "company_id", "company_name", "opportunity_id", "opportunity_name",
    "company_sector", "normalized_sector", "opportunity_sector",
    "sector_similarity", "profile_similarity", "product_similarity",
    "value_chain_score", "investment_readiness_score", "strategic_fit_score",
    "localization_score", "ai_score", "confidence_score", "decision",
    "final_score", "rank", "strengths", "risks", "recommended_engagement",
    "suggested_localization_model", "match_reason", "executive_summary",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-openai", action="store_true")
    ap.add_argument("--require-openai", action="store_true")
    ap.add_argument("--embed-blend", type=float, default=0.3)
    ap.add_argument("--chat-provider", choices=["auto", "azure", "public"], default="auto")
    ap.add_argument("--gpt-votes", type=int, default=3)
    ap.add_argument("--no-escalate", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--top-n", type=int, default=3, help="gate depth per opportunity")
    ap.add_argument("--narrative-top", type=int, default=5, help="narrative rows per company")
    ap.add_argument("--no-narratives", action="store_true")
    ap.add_argument("--weights", default=None, help="JSON file overriding scoring weights")
    ap.add_argument("--human-reviews", default="Data/human_reviews.csv")
    ap.add_argument("--env-file", default=None)
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(".env")
        if args.env_file:
            load_dotenv(args.env_file, override=True)
    except ImportError:
        pass

    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        with open(args.weights) as fh:
            weights.update(json.load(fh))
    wsum = sum(weights.values())
    weights = {k: v / wsum for k, v in weights.items()}

    backends = resolve_backends(args)
    chat_client, chat_models = backends["chat_client"], backends["chat_models"]
    if chat_client is None:
        sys.exit("FATAL: v3 requires a chat backend (enrichment + gate).")

    companies = load_companies()
    opps = load_opportunities()
    human = load_human_reviews(args.human_reviews)
    print(f"Loaded {len(companies)} companies, {len(opps)} opportunities, "
          f"{len(human)} analyst verdicts.")

    prof_mat, prod_mat, opp_mat, mode = build_vectors(companies, opps, args, backends)
    print(f"Embedding backend: {mode.upper()}")
    from matching_v2 import cosine_matrix
    sim_profile = cosine_matrix(prof_mat, opp_mat)
    sim_product = cosine_matrix(prod_mat, opp_mat)
    pct_profile = percentile_rank(sim_profile)
    pct_product = percentile_rank(sim_product)
    spec_profile = percentile_rank(sim_profile - sim_profile.mean(axis=1, keepdims=True))
    spec_product = percentile_rank(sim_product - sim_product.mean(axis=1, keepdims=True))
    sem_profile = (1 - SPECIFICITY_BLEND) * pct_profile + SPECIFICITY_BLEND * spec_profile
    sem_product = (1 - SPECIFICITY_BLEND) * pct_product + SPECIFICITY_BLEND * spec_product

    print("Enriching companies and opportunities (cached GPT classification)...")
    comp_enrich, opp_enrich = enrich_all(chat_client, chat_models, companies, opps,
                                         workers=args.workers)

    opp_names = opps["What is the opportunity name?"].tolist()
    rows = []
    for i, comp in companies.iterrows():
        ce = comp_enrich.get(comp["company_name"], {})
        c_node = ce.get("normalized_sector") or normalize_sector_label(comp["Sector"])
        c_role = ce.get("value_chain_role", "")
        c_role2 = ce.get("secondary_role", "")
        dims = ce.get("readiness", {}) or {}
        readiness, strategic, localization = compose_readiness(dims)
        class_conf = float(ce.get("classification_confidence", 0.3) or 0.3)
        comp_len = len(str(comp["company_profile"])) + len(str(comp["product and Services"]))

        for j, opp in opps.iterrows():
            oe = opp_enrich.get(opp_names[j], {})
            o_node = oe.get("normalized_sector") or normalize_sector_label(opp["Sector"])
            required = oe.get("required_roles", []) or []
            s_sim = sector_similarity(c_node, o_node)
            vc = value_chain_score(required, c_role, c_role2)
            p_sem = float(sem_profile[i, j])
            pr_sem = float(sem_product[i, j])
            base = (weights["sector"] * s_sim + weights["profile"] * p_sem
                    + weights["product"] * pr_sem + weights["value_chain"] * vc
                    + weights["readiness"] * readiness
                    + weights["strategic"] * strategic
                    + weights["localization"] * localization)
            factor, applied = compute_penalties(s_sim, p_sem, pr_sem, vc, c_role, required)
            final = round(max(0.05, base * factor), 3)
            rows.append({
                "company_id": i + 1, "company_name": comp["company_name"],
                "opportunity_id": j + 1, "opportunity_name": opp_names[j],
                "company_sector": comp["Sector"], "normalized_sector": c_node,
                "opportunity_sector": opp["Sector"],
                "sector_similarity": round(s_sim, 3),
                "profile_similarity": round(p_sem, 3),
                "product_similarity": round(pr_sem, 3),
                "value_chain_score": vc,
                "investment_readiness_score": readiness,
                "strategic_fit_score": strategic,
                "localization_score": localization,
                "final_score": final,
                "_penalties": ";".join(applied),
                "_class_conf": class_conf, "_comp_len": comp_len,
                "_role": c_role, "_required": required, "_i": i, "_j": j,
            })

    df = pd.DataFrame(rows)
    df["rank"] = (df.groupby("company_id")["final_score"]
                  .rank(method="first", ascending=False).astype(int))
    df["rank_for_opp"] = (df.groupby("opportunity_id")["final_score"]
                          .rank(method="first", ascending=False).astype(int))

    # graded gate on top-N per opportunity (reuses v2 gate + exemplars + drift)
    exemplar_pairs = build_exemplar_lines(human, companies)
    prior = load_prior_verdicts()
    df["gate"] = ""
    df["gate_agreement"] = ""
    todo = df[df["rank_for_opp"] <= args.top_n]
    print(f"Gate: validating {len(todo)} pairs ({args.gpt_votes}-vote)...")

    def _gate(item):
        idx, row = item
        lines = [l for p, l in exemplar_pairs
                 if p != (row["company_name"], row["opportunity_name"])][-8:]
        return idx, gpt_validate(chat_client, chat_models,
                                 companies.loc[row["_i"]], opps.loc[row["_j"]],
                                 votes=args.gpt_votes, escalate=not args.no_escalate,
                                 exemplars="\n".join(lines))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for idx, (fit, conf, expl, model, agree) in ex.map(_gate, list(todo.iterrows())):
            df.at[idx, "gate"] = fit
            df.at[idx, "gate_agreement"] = agree

    labels = [{"ts": datetime.now(timezone.utc).isoformat(),
               "company": r.company_name, "opportunity": r.opportunity_name,
               "decision": r.gate, "confidence": 0.0, "agreement": r.gate_agreement,
               "model": "v3", "votes": args.gpt_votes, "rubric": RUBRIC_HASH,
               "final_score": r.final_score, "embed_mode": mode}
              for r in df.itertuples() if r.gate]
    if labels:
        os.makedirs(os.path.dirname(LABELS_JSONL), exist_ok=True)
        with open(LABELS_JSONL, "a") as fh:
            for item in labels:
                fh.write(json.dumps(item) + "\n")
    flips = sum(1 for r in df.itertuples() if r.gate and
                prior.get((r.company_name, r.opportunity_name), {}).get("decision")
                not in ("", None, r.gate))
    print(f"Gate verdicts recorded: {len(labels)}; changed vs prior: {flips}.")

    # decisions, confidence
    df["human_verdict"] = [
        {1: "Agree", 0: "Disagree"}.get(human.get((r.company_name, r.opportunity_name), -1), "")
        for r in df.itertuples()]
    decisions, confidences, ai_scores = [], [], []
    # iterrows, not itertuples: underscore-prefixed helper columns are renamed
    # positionally by itertuples and become unreachable by name.
    for _, r in df.iterrows():
        d = decide(r["final_score"], r["gate"], r["human_verdict"], bool(r["gate"]))
        comps = [r["sector_similarity"], r["profile_similarity"], r["product_similarity"],
                 r["value_chain_score"], r["investment_readiness_score"]]
        c = confidence_score(r["_comp_len"], 1500, r["_class_conf"], comps,
                             r["gate_agreement"])
        decisions.append(d)
        confidences.append(f"{c} ({confidence_label(c)})")
        ai_scores.append(1 if TIER_ORDER.index(d) <= TIER_ORDER.index("Good Match") else 0)
    df["decision"] = decisions
    df["confidence_score"] = confidences
    df["ai_score"] = ai_scores

    # narratives for the export slice
    out_rows = df[df["rank"] <= args.narrative_top].copy()
    for k in ["strengths", "risks", "recommended_engagement",
              "suggested_localization_model", "match_reason", "executive_summary"]:
        out_rows[k] = ""
    if not args.no_narratives:
        print(f"Narratives for {len(out_rows)} rows...")
        done = [0]

        def _narr(item):
            idx, row = item
            g = generate_narrative(chat_client, chat_models,
                                   companies.loc[row["_i"]], opps.loc[row["_j"]],
                                   row["decision"], row["_role"], row["_required"])
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(out_rows):
                print(f"  {done[0]}/{len(out_rows)}", flush=True)
            return idx, g

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for idx, g in ex.map(_narr, list(out_rows.iterrows())):
                for k, v in g.items():
                    out_rows.at[idx, k] = v

    # Deliverable ordering: best decisions first, then score - the file opens
    # on the recommendations, not on an alphabetical wall of Potential rows.
    out_rows["_tier"] = out_rows["decision"].map({t: i for i, t in enumerate(TIER_ORDER)})
    out = out_rows.sort_values(["_tier", "final_score"],
                               ascending=[True, False])[COLUMNS]
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    dist = out["decision"].value_counts().to_dict()
    print(f"\nWrote {OUTPUT_CSV}: {len(out)} rows.")
    print("Decision distribution:", dist)
    print("Penalties applied on", int((df['_penalties'].str.len() > 0).sum()),
          "of", len(df), "pairs.")
    return out


if __name__ == "__main__":
    main()
