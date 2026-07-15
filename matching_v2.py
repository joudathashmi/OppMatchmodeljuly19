#!/usr/bin/env python3
"""
Company ↔ Opportunity matching — v2.

Rebuilt from Code.ipynb with the following fixes and upgrades:

  FIX 1  Sector vocabulary is protected from dynamic stopword learning.
         (v1 silently stripped "industrial"/"manufacturing" and eliminated
         36/64 companies from every match.)
  FIX 2  Semantic scores are percentile-normalized within the run, so
         calibration is independent of the embedding backend (OpenAI vs
         TF-IDF fallback produce comparable bands).
  FIX 3  One scoring pass; both ranking views (per-opportunity and
         per-company) are derived from the same table. Halves compute and
         GPT spend vs v1's duplicated directions.
  FIX 4  Vectorized cosine similarity (matrix ops, not per-pair calls).
  FIX 5  Batched + cached embeddings (v1 made one API call per text).
  FIX 6  The embedding mode is stamped on every output row and the run
         fails loudly with --require-openai if the API is unavailable.
  FIX 7  Abstention: an opportunity with no qualified candidate says so,
         instead of force-ranking the least-bad company.
  FIX 8  Popularity correction: a company's score is blended with its
         specificity (how much this pair beats that company's own average),
         so long generic profiles stop winning every opportunity.
  FIX 9  GPT validation runs once, after scores are frozen, on the final
         top-N; its verdict gates the decision label and is stored in its
         own columns — it never overwrites the score scale. Verdicts are
         appended to Output/gpt_labels.jsonl to build an evaluation set.

Usage:
  python3 matching_v2.py                 # auto mode (OpenAI if key works)
  python3 matching_v2.py --no-gpt        # skip GPT validation
  python3 matching_v2.py --require-openai  # fail instead of TF-IDF fallback

Inputs : Data/companies.xlsx, Data/new_opportunities.xlsx
Outputs: Output/matches_v2.xlsx (+ Output/gpt_labels.jsonl)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------- configuration -----------------------------

DATA_COMPANIES = "Data/companies.xlsx"
DATA_OPPORTUNITIES = "Data/new_opportunities.xlsx"
OUTPUT_XLSX = "Output/matches_v2.xlsx"
LABELS_JSONL = "Output/gpt_labels.jsonl"
EMB_CACHE = "Output/emb_cache_v2.npz"

EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODELS = ["gpt-4.1", "gpt-4o", "gpt-4o-mini"]
GPT_TOP_N_PER_OPPORTUNITY = 3
EMBED_BATCH = 96

# final score weights (all components are 0-1)
W_PROFILE = 0.30
W_PRODUCT = 0.30
W_SECTOR = 0.25
W_EVIDENCE = 0.15

# semantic = blend of global percentile and company-specificity percentile
SPECIFICITY_BLEND = 0.35

# qualification (absolute, not percentile — drives abstention)
MIN_SECTOR_SCORE = 0.35
MIN_EVIDENCE_TERMS = 2
SOFT_MATCH_MIN_PCT = 0.60  # semantic percentile needed to soft-pass sector

STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this", "are", "was",
    "were", "will", "can", "their", "its", "our", "your", "about", "over",
    "under", "what", "when", "where", "who", "why", "how", "is", "of", "to",
    "in", "on", "a", "an",
}

GENERIC_BUSINESS_TERMS = {
    "advanced", "company", "companies", "group", "global", "regional",
    "international", "development", "develop", "developing", "solution",
    "solutions", "service", "services", "market", "markets", "business",
    "industry", "industries", "portfolio", "capabilities", "operations",
    "operational", "based", "including", "providing", "provide", "provides",
    "supported", "support", "value", "quality", "efficient", "efficiency",
    "strategic", "innovation", "innovative", "growth", "project", "projects",
    "systems", "products", "product", "clients", "customer", "customers",
    "cities", "which", "specific", "future", "life", "safe", "driving",
    "smart", "digital", "integrated", "operating", "sustainable", "heavy",
    "flow", "provision", "process",
}

SECTOR_TOKEN_MAP = {
    "information": "ict", "communication": "ict", "communications": "ict",
    "telecommunications": "telecom", "electrical": "electronics",
    "electronic": "electronics", "semiconductors": "semiconductor",
    "manufacture": "manufacturing", "industrials": "industrial",
    "pharmaceuticals": "pharma", "pharmaceutical": "pharma",
    "medicine": "medical", "biologics": "biotech", "health": "healthcare",
    "medtech": "medical",
}

SECTOR_GROUPS = [
    {"ict", "hardware", "software", "electronics", "digital", "semiconductor", "telecom"},
    {"medical", "healthcare", "biotech", "pharma"},
    {"manufacturing", "industrial", "factory", "engineering", "construction", "materials"},
    {"energy", "power", "renewable", "oil", "gas", "utilities", "water"},
    {"mining", "minerals", "metals"},
]

SECTOR_ONTOLOGY = {
    "ict": {"telecom", "network", "fiber", "datacenter", "server", "cloud",
            "embedded", "electronics", "hardware", "software", "semiconductor"},
    "telecom": {"5g", "smallcell", "macro", "baseband", "antenna", "ran", "backhaul"},
    "medical": {"diagnostic", "imaging", "clinical", "device", "devices", "sterile"},
    "pharma": {"biotech", "api", "formulation", "fillfinish", "gmp", "biologics"},
    "manufacturing": {"assembly", "fabrication", "machining", "tooling", "automation", "line"},
    "industrial": {"automation", "controls", "scada", "instrumentation", "maintenance"},
    "energy": {"grid", "storage", "renewable", "solar", "wind", "utility", "power"},
}

BRIDGE_RULES = [
    # (side_a sectors, side_b sectors, capability terms, score, name)
    ({"industrial", "manufacturing"}, {"ict", "hardware", "electronics", "telecom"},
     {"electronics", "electrical", "component", "components", "assembly", "hardware",
      "cables", "cabling", "wiring", "automation", "control", "sensor", "sensors",
      "pcb", "device", "devices", "enclosures"},
     0.58, "Industrial ↔ ICT"),
    ({"industrial", "manufacturing"}, {"medical", "pharma", "biotech", "healthcare"},
     {"precision", "diagnostic", "imaging", "sterile", "medical", "healthcare",
      "cleanroom", "instrument", "instruments", "biotech", "pharma", "chemical",
      "chemicals"},
     0.48, "Industrial ↔ Medical/Pharma"),
    ({"ict", "hardware", "electronics"}, {"medical", "pharma", "healthcare"},
     {"medical", "healthcare", "diagnostic", "imaging", "device", "devices",
      "data", "software"},
     0.52, "ICT ↔ Medical"),
    ({"energy", "oil", "gas", "utilities", "mining"}, {"pharma", "medical", "biotech"},
     {"chemical", "chemicals", "specialty", "processing", "refining", "synthesis"},
     0.45, "Energy/Chemicals ↔ Pharma"),
]

DYNAMIC_COMMON_RATIO = 0.12
DYNAMIC_COMMON_MIN_DOCS = 5

# geography / market words that show up in opportunity market-data text but
# say nothing about capability fit — excluded from evidence terms
NON_CAPABILITY_TERMS = {
    "saudi", "arabia", "riyadh", "jeddah", "dammam", "gulf", "gcc", "mena",
    "middle", "east", "asia", "europe", "africa", "america", "united",
    "states", "china", "india", "germany", "national", "kingdom", "vision",
    "2030", "prices", "price", "cost", "costs", "demand", "supply", "export",
    "exports", "import", "imports", "billion", "million", "cagr", "forecast",
    "sabic", "aramco", "neom",
}
MIN_EVIDENCE_IDF = 1.5

# ------------------------------- text utils ------------------------------


def preprocess(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_static(text) -> set:
    return {
        t for t in preprocess(text).split()
        if t not in STOPWORDS and t not in GENERIC_BUSINESS_TERMS
        and len(t) > 2 and not t.isdigit()
    }


class Vocabulary:
    """Corpus-learned common tokens + protected sector vocabulary (FIX 1)."""

    def __init__(self):
        self.common: set = set()
        self.protected: set = set()
        self.idf: dict = {}
        self.domain: set = set()

    def fit(self, corpus: list, sector_texts: list, extra_domain_texts: list):
        self.protected = set()
        for val in sector_texts:
            self.protected |= tokenize_static(val)
        for key, vals in SECTOR_ONTOLOGY.items():
            self.protected |= {key} | set(vals)
        for _, _, terms, _, _ in BRIDGE_RULES:
            self.protected |= set(terms)

        doc_counts: dict = {}
        for text in corpus:
            for tok in tokenize_static(text):
                doc_counts[tok] = doc_counts.get(tok, 0) + 1
        n = max(1, len(corpus))
        self.common = {
            tok for tok, cnt in doc_counts.items()
            if cnt >= DYNAMIC_COMMON_MIN_DOCS and cnt / n >= DYNAMIC_COMMON_RATIO
            and tok not in self.protected  # <-- the v1 bug fix
        }
        self.idf = {tok: float(np.log(n / (1 + cnt))) + 1.0 for tok, cnt in doc_counts.items()}

        self.domain = set(self.protected)
        for text in extra_domain_texts:
            self.domain |= tokenize_static(text)
        self.domain = {t for t in self.domain if len(t) >= 4}

    def tokenize(self, text) -> set:
        return {t for t in tokenize_static(text) if t not in self.common}


VOCAB = Vocabulary()

# ------------------------------- embeddings -------------------------------


def _hash(text: str) -> str:
    return hashlib.md5(f"{EMBEDDING_MODEL}::{text}".encode()).hexdigest()


def embed_texts(texts: list, client) -> np.ndarray:
    """Batched OpenAI embeddings with an on-disk cache (FIX 5)."""
    cache = {}
    if os.path.exists(EMB_CACHE):
        loaded = np.load(EMB_CACHE)
        cache = {k: loaded[k] for k in loaded.files}

    missing = [t for t in texts if _hash(t) not in cache]
    for start in range(0, len(missing), EMBED_BATCH):
        chunk = missing[start:start + EMBED_BATCH]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
        for text, item in zip(chunk, resp.data):
            cache[_hash(text)] = np.asarray(item.embedding, dtype=np.float32)
    if missing:
        os.makedirs(os.path.dirname(EMB_CACHE), exist_ok=True)
        np.savez_compressed(EMB_CACHE, **cache)

    return np.vstack([cache[_hash(t)] for t in texts])


def build_vectors(companies: pd.DataFrame, opps: pd.DataFrame, args):
    """Returns (profile_mat, product_mat, opp_mat, mode)."""
    corpus = (
        companies["combined"].tolist()
        + companies["products_clean"].tolist()
        + opps["requirement"].tolist()
    )
    client = None
    if not args.no_openai and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            client.embeddings.create(model=EMBEDDING_MODEL, input=["ping"])  # auth check
        except Exception as e:
            client = None
            msg = f"OpenAI embeddings UNAVAILABLE ({type(e).__name__}): falling back to TF-IDF."
            if args.require_openai:
                sys.exit(f"FATAL: {msg} (--require-openai set)")
            print(f"\n{'!' * 70}\n{msg}\nScores are NOT comparable with OpenAI-mode runs.\n{'!' * 70}\n")
    elif args.require_openai:
        sys.exit("FATAL: OPENAI_API_KEY not set and --require-openai given.")

    if client is not None:
        prof = embed_texts(companies["combined"].tolist(), client)
        prod = embed_texts(companies["products_clean"].tolist(), client)
        opp = embed_texts(opps["requirement"].tolist(), client)
        return prof, prod, opp, "openai", client

    # Fallback: hybrid word + character TF-IDF (stronger than v1's word-only)
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    word.fit(corpus)
    char.fit(corpus)

    def vec(texts):
        return hstack([word.transform(texts), 0.5 * char.transform(texts)]).toarray()

    return (vec(companies["combined"]), vec(companies["products_clean"]),
            vec(opps["requirement"]), "tfidf", None)


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    Bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return An @ Bn.T


def percentile_rank(m: np.ndarray) -> np.ndarray:
    flat = m.flatten()
    order = flat.argsort().argsort().astype(float)
    return (order / max(1, len(flat) - 1)).reshape(m.shape)

# ----------------------------- sector scoring -----------------------------


def canon(tokens: set) -> set:
    return {SECTOR_TOKEN_MAP.get(t, t) for t in tokens}


def expand(tokens: set) -> set:
    out = set(tokens)
    for t in tokens:
        out |= SECTOR_ONTOLOGY.get(t, set())
    return out


def sector_score(company_sector: str, opp_sector: str, capability_tokens: set):
    """Returns (score, label, reason, bridge_name)."""
    c = expand(canon(VOCAB.tokenize(company_sector)))
    o = expand(canon(VOCAB.tokenize(opp_sector)))
    if not c or not o:
        return 0.0, "Unknown", "Sector text missing or uninformative.", None
    if preprocess(company_sector) == preprocess(opp_sector):
        return 1.0, "Exact", f"Exact sector match: '{company_sector}'.", None

    overlap = c & o
    jaccard = len(overlap) / len(c | o)
    group_hit = any((c & g) and (o & g) for g in SECTOR_GROUPS)
    score = max(jaccard, 0.75 if group_hit else 0.0)

    reason = (
        f"Shared sector terms: {', '.join(sorted(overlap)[:5])}." if overlap
        else "Sector families align (same broader industry group)." if group_hit
        else f"No direct sector overlap ('{company_sector}' vs '{opp_sector}')."
    )
    bridge_name = None
    if score < 0.50:
        for side_a, side_b, terms, bscore, name in BRIDGE_RULES:
            crossed = (c & side_a and o & side_b) or (c & side_b and o & side_a)
            hits = sorted(capability_tokens & terms)
            if crossed and hits and bscore > score:
                score = bscore
                bridge_name = name
                reason = f"Cross-sector bridge ({name}) via capabilities: {', '.join(hits[:6])}."
                break

    label = ("Strong" if score >= 0.80 else "Moderate" if score >= 0.50
             else "Weak" if score > 0 else "No")
    return score, label, reason, bridge_name

# --------------------------- evidence + fusion ----------------------------


def domain_overlap(company_text: str, opp_text: str) -> list:
    """Domain-vocabulary terms shared by both sides, ranked by IDF.

    Geographic/market words and corpus-frequent terms are excluded — evidence
    must be capability vocabulary, not shared boilerplate.
    """
    shared = VOCAB.tokenize(company_text) & VOCAB.tokenize(opp_text) & VOCAB.domain
    shared = {
        t for t in shared
        if t not in NON_CAPABILITY_TERMS and VOCAB.idf.get(t, 0.0) >= MIN_EVIDENCE_IDF
    }
    return sorted(shared, key=lambda t: VOCAB.idf.get(t, 0.0), reverse=True)


def evidence_score(terms: list) -> float:
    """Absolute 0-1 evidence strength from IDF mass of shared domain terms."""
    total = sum(VOCAB.idf.get(t, 0.0) for t in terms[:10])
    return float(1.0 - np.exp(-total / 8.0))


def decision_label(final: float, qualified: bool, real_sector_link: bool,
                   gpt_decision=None, bridge_only: bool = False) -> str:
    """Business label. High/Good Fit require a real sector relationship
    (exact/family/bridge); soft sector matches cap at Review Needed so an
    analyst confirms before anything client-facing.

    A match whose only sector link is a cross-sector bridge (bridge_only) is
    inherently more speculative than a same-family match, so on model score
    alone it caps at "Good Fit" — only a GPT "Yes" can elevate it to High Fit.
    """
    if gpt_decision == "No" or not qualified:
        return "Low Fit"
    if gpt_decision == "Yes":
        return "High Fit" if final >= 0.60 else "Good Fit"
    if not real_sector_link:
        return "Review Needed" if final >= 0.55 else "Low Fit"
    if bridge_only:
        return ("Good Fit" if final >= 0.70
                else "Review Needed" if final >= 0.50 else "Low Fit")
    if final >= 0.85:
        return "High Fit"
    if final >= 0.70:
        return "Good Fit"
    if final >= 0.50:
        return "Review Needed"
    return "Low Fit"

# ------------------------------ GPT validation ----------------------------


def gpt_validate(client, comp: pd.Series, opp: pd.Series):
    prompt = f"""
Evaluate company-opportunity fit and return strict JSON:
{{"decision": "Yes or No", "confidence": 0.0 to 1.0,
  "explanation": "2-4 sentences on sector, profile and product fit"}}

Company Name: {comp['company_name']}
Company Sector: {comp['Sector']}
Company Profile: {comp['company_profile']}
Products/Services: {comp['product and Services']}

Opportunity Name: {opp['What is the opportunity name?']}
Opportunity Sector: {opp['Sector']}
Description: {opp['What is the opportunity description?']}
Highlights: {opp['What are the investment highlights?']}
Demand Drivers: {opp['What are the key demand drivers?']}
Required Materials: {opp['What materials are involved or required in the project?']}
"""
    for model in GPT_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0,
                messages=[
                    {"role": "system", "content": "You are a strict industrial matching analyst."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            content = re.sub(r"^```(?:json)?|```$", "", content).strip()
            parsed = json.loads(content)
            decision = "Yes" if str(parsed.get("decision", "No")).strip().lower().startswith("y") else "No"
            conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
            return decision, conf, str(parsed.get("explanation", "")).strip(), model
        except Exception:
            continue
    return "Not Run", 0.0, "GPT validation unavailable.", None

# --------------------------------- loaders --------------------------------


def load_companies() -> pd.DataFrame:
    df = pd.read_excel(DATA_COMPANIES)
    df = df.rename(columns={
        "Company Name": "company_name", "Company Profile": "company_profile",
        "Product/Services": "product and Services",
    })
    required = ["company_name", "company_profile", "product and Services", "Sector"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"companies.xlsx missing columns: {missing}")
    dupes = df["company_name"][df["company_name"].duplicated()].tolist()
    if dupes:
        print(f"WARNING: duplicate company names (keeping all, matched independently): {dupes}")
    df["combined"] = (
        df[["company_name", "company_profile", "product and Services"]]
        .astype(str).agg(" ".join, axis=1).apply(preprocess)
    )
    df["products_clean"] = df["product and Services"].astype(str).apply(preprocess)
    return df.reset_index(drop=True)


def load_opportunities() -> pd.DataFrame:
    df = pd.read_excel(DATA_OPPORTUNITIES)
    fields = [
        "What is the opportunity name?", "What is the opportunity description?",
        "What are the investment highlights?",
        "What is the value proposition of this opportunity?",
        "What are the key demand drivers?",
        "What materials are involved or required in the project?",
        "Who are the key players in this sector or project?",
        "Market data", "Risks and mitigations",
    ]
    missing = [c for c in fields[:6] + ["Sector"] if c not in df.columns]
    if missing:
        raise KeyError(f"new_opportunities.xlsx missing columns: {missing}")
    df["requirement"] = df.apply(
        lambda r: preprocess(" ".join(str(r.get(f, "")) for f in fields)), axis=1
    )
    return df.reset_index(drop=True)

# ----------------------------------- main ----------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gpt", action="store_true", help="skip GPT validation")
    ap.add_argument("--no-openai", action="store_true", help="force TF-IDF fallback")
    ap.add_argument("--require-openai", action="store_true",
                    help="fail hard instead of falling back to TF-IDF")
    ap.add_argument("--top-n", type=int, default=GPT_TOP_N_PER_OPPORTUNITY)
    args = ap.parse_args()

    companies = load_companies()
    opps = load_opportunities()
    print(f"Loaded {len(companies)} companies, {len(opps)} opportunities.")

    corpus = (companies["combined"].tolist() + companies["products_clean"].tolist()
              + opps["requirement"].tolist())
    sector_texts = companies["Sector"].astype(str).tolist() + opps["Sector"].astype(str).tolist()
    extra_domain = opps["Sector"].astype(str).tolist() + opps[
        "What materials are involved or required in the project?"
    ].astype(str).tolist()
    VOCAB.fit(corpus, sector_texts, extra_domain)
    print(f"Vocabulary: {len(VOCAB.common)} corpus-common tokens suppressed, "
          f"{len(VOCAB.protected)} sector tokens protected, "
          f"{len(VOCAB.domain)} domain keywords.")

    prof_mat, prod_mat, opp_mat, mode, client = build_vectors(companies, opps, args)
    print(f"Embedding mode: {mode.upper()}")

    sim_profile = cosine_matrix(prof_mat, opp_mat)
    sim_product = cosine_matrix(prod_mat, opp_mat)
    pct_profile = percentile_rank(sim_profile)
    pct_product = percentile_rank(sim_product)

    # FIX 8 — specificity: how much a pair beats the company's own average
    spec_profile = percentile_rank(sim_profile - sim_profile.mean(axis=1, keepdims=True))
    spec_product = percentile_rank(sim_product - sim_product.mean(axis=1, keepdims=True))
    sem_profile = (1 - SPECIFICITY_BLEND) * pct_profile + SPECIFICITY_BLEND * spec_profile
    sem_product = (1 - SPECIFICITY_BLEND) * pct_product + SPECIFICITY_BLEND * spec_product

    rows = []
    for i, comp in companies.iterrows():
        cap_tokens = expand(canon(VOCAB.tokenize(
            f"{comp['company_profile']} {comp['product and Services']}"
        )))
        for j, opp in opps.iterrows():
            s_score, s_label, s_reason, bridge = sector_score(
                comp["Sector"], opp["Sector"], cap_tokens
            )
            terms = domain_overlap(
                f"{comp['company_profile']} {comp['product and Services']}",
                opp["requirement"],
            )
            ev = evidence_score(terms)

            # soft sector pass: no sector overlap and no bridge, only semantic +
            # domain-evidence signal. Surfaced for transparency but NOT qualified
            # — pure semantic similarity (especially in TF-IDF mode) is too noisy
            # to stand in for a real sector relationship, and was producing
            # nonsense top picks (e.g. a transformer maker topping an MRI opp).
            soft_match = False
            if (s_label in ("No", "Unknown") and len(terms) >= MIN_EVIDENCE_TERMS
                    and max(sem_profile[i, j], sem_product[i, j]) >= SOFT_MATCH_MIN_PCT
                    and s_score < 0.35):
                s_score, s_label = 0.35, "Weak"
                soft_match = True
                s_reason = (f"Soft candidate only: sector labels differ and no "
                            f"cross-sector bridge fired; shared domain terms "
                            f"({', '.join(terms[:4])}) are semantic signal, not a "
                            f"sector relationship. Not qualified.")

            real_sector_link = s_label in ("Exact", "Strong", "Moderate") or bridge is not None
            bridge_only = bridge is not None and s_label not in ("Exact", "Strong", "Moderate")
            qualified = (not soft_match and s_score >= MIN_SECTOR_SCORE
                         and len(terms) >= MIN_EVIDENCE_TERMS)
            final = (W_PROFILE * sem_profile[i, j] + W_PRODUCT * sem_product[i, j]
                     + W_SECTOR * s_score + W_EVIDENCE * ev)
            label = decision_label(final, qualified, real_sector_link,
                                   bridge_only=bridge_only)

            rows.append({
                "company": comp["company_name"], "company_sector": comp["Sector"],
                "opportunity": opp["What is the opportunity name?"],
                "opportunity_sector": opp["Sector"],
                "raw_profile_cosine": round(float(sim_profile[i, j]), 4),
                "raw_product_cosine": round(float(sim_product[i, j]), 4),
                "semantic_profile": round(float(sem_profile[i, j]), 3),
                "semantic_product": round(float(sem_product[i, j]), 3),
                "sector_score": round(s_score, 3), "sector_label": s_label,
                "bridge": bridge or "", "evidence_score": round(ev, 3),
                "evidence_terms": ", ".join(terms[:8]),
                "n_evidence_terms": len(terms),
                "qualified": qualified, "real_sector_link": real_sector_link,
                "bridge_only": bridge_only, "soft_candidate": soft_match,
                "final_score": round(final, 3),
                "ai_decision": label, "sector_reason": s_reason,
                "embed_mode": mode,
                "_i": i, "_j": j,
            })

    df = pd.DataFrame(rows)

    # Rank among QUALIFIED pairs only. Unqualified pairs (soft candidates,
    # no-evidence pairs) get no rank — otherwise a high-scoring but unqualified
    # soft candidate would hold rank 1 and push genuine bridged candidates past
    # the view's rank cutoff, making them silently disappear.
    def qualified_rank(group_col: str) -> pd.Series:
        r = pd.Series(np.nan, index=df.index)
        qmask = df["qualified"]
        r[qmask] = (df[qmask].groupby(group_col)["final_score"]
                    .rank(method="first", ascending=False))
        return r

    df["rank_for_opportunity"] = qualified_rank("opportunity")
    df["rank_for_company"] = qualified_rank("company")

    # FIX 9 — GPT validation on frozen scores, qualified top-N only
    df["gpt_decision"] = ""
    df["gpt_confidence"] = np.nan
    df["gpt_explanation"] = ""
    if not args.no_gpt and client is not None:
        todo = df[(df["rank_for_opportunity"] <= args.top_n) & df["qualified"]]
        print(f"GPT-validating {len(todo)} qualified top-{args.top_n} pairs...")
        labels = []
        for idx, row in todo.iterrows():
            decision, conf, expl, model = gpt_validate(
                client, companies.loc[row["_i"]], opps.loc[row["_j"]]
            )
            df.at[idx, "gpt_decision"] = decision
            df.at[idx, "gpt_confidence"] = conf
            df.at[idx, "gpt_explanation"] = expl
            df.at[idx, "ai_decision"] = decision_label(
                row["final_score"], row["qualified"], row["real_sector_link"],
                gpt_decision=decision, bridge_only=row["bridge_only"],
            )
            labels.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "company": row["company"], "opportunity": row["opportunity"],
                "decision": decision, "confidence": conf, "model": model,
                "final_score": row["final_score"], "embed_mode": mode,
            })
        if labels:
            os.makedirs(os.path.dirname(LABELS_JSONL), exist_ok=True)
            with open(LABELS_JSONL, "a") as fh:
                for item in labels:
                    fh.write(json.dumps(item) + "\n")
            print(f"Appended {len(labels)} verdicts to {LABELS_JSONL}")
    elif not args.no_gpt:
        print("GPT validation skipped: no working OpenAI client.")

    df = df.drop(columns=["_i", "_j"])

    # FIX 7 — abstention report
    abstained = []
    for opp_name, grp in df.groupby("opportunity"):
        if not grp["qualified"].any():
            abstained.append({
                "opportunity": opp_name,
                "status": "No qualified candidate",
                "best_unqualified": grp.sort_values("final_score", ascending=False).iloc[0]["company"],
            })
    abstain_df = pd.DataFrame(abstained)

    opp_view = df[df["qualified"] & (df["rank_for_opportunity"] <= max(args.top_n, 5))].sort_values(
        ["opportunity", "rank_for_opportunity"]
    )
    comp_view = df[df["qualified"] & (df["rank_for_company"] <= 3)].sort_values(
        ["company", "rank_for_company"]
    )
    diag = pd.DataFrame([
        {"metric": "run_timestamp_utc", "value": datetime.now(timezone.utc).isoformat()},
        {"metric": "embedding_mode", "value": mode},
        {"metric": "pairs_total", "value": len(df)},
        {"metric": "pairs_qualified", "value": int(df["qualified"].sum())},
        {"metric": "companies_total", "value": len(companies)},
        {"metric": "companies_with_qualified_match",
         "value": int(df[df["qualified"]]["company"].nunique())},
        {"metric": "opportunities_abstained", "value": len(abstain_df)},
        {"metric": "median_raw_profile_cosine", "value": round(float(df["raw_profile_cosine"].median()), 4)},
        {"metric": "top3_company_concentration",
         "value": round(df[df["rank_for_opportunity"] <= 3]["company"].value_counts(normalize=True).max(), 3)},
    ])

    os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        opp_view.to_excel(writer, sheet_name="Opportunity_View", index=False)
        comp_view.to_excel(writer, sheet_name="Company_View", index=False)
        df.sort_values(["opportunity", "rank_for_opportunity"]).to_excel(
            writer, sheet_name="All_Pairs", index=False)
        (abstain_df if len(abstain_df) else pd.DataFrame([{"opportunity": "-", "status": "All opportunities have qualified candidates"}])).to_excel(
            writer, sheet_name="Abstentions", index=False)
        diag.to_excel(writer, sheet_name="Diagnostics", index=False)

    print(f"\nSaved {OUTPUT_XLSX}")
    print(diag.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
