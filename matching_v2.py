#!/usr/bin/env python3
"""
Company <-> Opportunity matching.

Pipeline (see docs/framework.md for the eight-step overview):
  1. Preprocess and focus company/opportunity text (boilerplate stripped).
  2. Sector ontology expansion; protected sector vocabulary.
  3. Sector scoring with cross-sector bridges (no hard filtering); abstention.
  4. Semantic embedding (text-embedding-3-large) with a capability-focused blend;
     percentile-normalized cosine; specificity/popularity correction.
  5. Product/service matching; weighted fusion into final_score.
  6. Graded GPT gate (Direct / Partial / None) with self-consistency voting;
     verdict governs the business label and is persisted to gpt_labels.jsonl.
  7. GPT-aware abstention; unvalidated pairs cannot claim Good/High Fit.
  8. Ranking and export.

Backends: public OpenAI and/or Azure OpenAI, resolved independently for chat and
embeddings (see resolve_backends). Clients carry a request timeout and retries.

Usage:
  python3 matching_v2.py                    # public gpt-4.1 gate, focus blend on
  python3 matching_v2.py --no-gpt           # scores only
  python3 matching_v2.py --require-openai   # fail instead of TF-IDF fallback
  python3 matching_v2.py --chat-provider azure   # gate in-tenant on Azure
  python3 matching_v2.py --embed-blend 0    # disable the focus blend

Inputs : Data/companies.xlsx, Data/new_opportunities.xlsx
Outputs: Output/matches_v2.xlsx (+ Output/gpt_labels.jsonl, Output/emb_cache_v2.npz)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional; env can be exported instead
    load_dotenv = None

# ----------------------------- configuration -----------------------------

DATA_COMPANIES = "Data/companies.xlsx"
DATA_OPPORTUNITIES = "Data/new_opportunities.xlsx"
OUTPUT_XLSX = "Output/matches_v2.xlsx"
LABELS_JSONL = "Output/gpt_labels.jsonl"
EMB_CACHE = "Output/emb_cache_v2.npz"
NEEDS_CACHE = "Output/needs_cache_v1.json"
# Human-in-the-loop: reviews exported from the GUI, dropped here by the analyst.
# Verdicts override the gate's own labels in calibration (gold beats silver).
HUMAN_REVIEWS_CSV = "Data/human_reviews.csv"
HUMAN_LABEL_WEIGHT = 3.0

# Consortium view readiness gate: an opportunity gets a needs breakdown only if
# GPT can extract at least this many needs whose supporting quotes VERIFY
# against the brief text. Below the bar the opportunity keeps the plain ranked
# view (fallback), so thin briefs never produce invented value chains.
MIN_VERIFIED_NEEDS = 3

EMBEDDING_MODEL = "text-embedding-3-large"
# gpt-4.1 first: the A/B test showed it is the most precise and stable gate
# (accepts only the single most defensible fit, at the highest confidence).
GPT_MODELS = ["gpt-4.1", "gpt-4o", "gpt-4o-mini"]
GPT_TOP_N_PER_OPPORTUNITY = 3
# Self-consistency: LLM verdicts are non-deterministic even at temperature 0, so
# the gate samples the model GPT_VOTES times and majority-votes. The agreement
# ratio becomes a calibrated confidence and is reported per pair.
GPT_VOTES = 3
GPT_TEMPERATURE = 0.3
GPT_WORKERS = 8            # concurrent gate validations
EMBED_BATCH = 96
AZURE_API_VERSION_DEFAULT = "2024-08-01-preview"
# A per-request timeout is essential: the SDK default is 600s, so a single
# stalled connection can wedge a run. Bound each attempt and let the SDK retry.
OPENAI_TIMEOUT = 60.0
OPENAI_RETRIES = 3

# Capability-focused embeddings (embedding path only). We embed both the full
# text and a version with corporate/price/geography boilerplate stripped, then
# blend the two vectors. Blending (rather than replacing) sharpens the semantic
# signal while keeping the match ordering stable: at 0.3 the full-vs-blend rank
# correlation is 0.96 and only ~10/64 companies change top match, vs 22/64 for a
# full replace. Set to 0.0 to disable (exact pre-focus behaviour).
EMBED_FOCUS_BLEND = 0.3

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

# Each rule: (side_a sectors, side_b sectors, capability terms, score, name, min_hits).
# min_hits is the number of DISTINCT capability terms a company must share for the
# bridge to fire. A single shared generic word ("chemical", "precision") is not a
# bridge — GPT rejected 100% of the single-term pharma/medtech bridges — so the
# weak cross-domain bridges require >=2 terms. `strong_terms` (when present) is a
# subset that must contribute at least one hit, so the required terms can't all be
# generic industrial vocabulary.
BRIDGE_RULES = [
    ({"industrial", "manufacturing"}, {"ict", "hardware", "electronics", "telecom"},
     {"electronics", "electrical", "component", "components", "assembly", "hardware",
      "cables", "cabling", "wiring", "automation", "control", "sensor", "sensors",
      "pcb", "device", "devices", "enclosures"},
     0.58, "Industrial ↔ ICT", 2),
    ({"industrial", "manufacturing"}, {"medical", "pharma", "biotech", "healthcare"},
     {"precision", "diagnostic", "imaging", "sterile", "medical", "healthcare",
      "cleanroom", "instrument", "instruments", "biotech", "pharma", "chemical",
      "chemicals"},
     0.48, "Industrial ↔ Medical/Pharma", 2),
    ({"ict", "hardware", "electronics"}, {"medical", "pharma", "healthcare"},
     {"medical", "healthcare", "diagnostic", "imaging", "device", "devices",
      "data", "software"},
     0.52, "ICT ↔ Medical", 2),
    ({"energy", "oil", "gas", "utilities", "mining"}, {"pharma", "medical", "biotech"},
     {"chemical", "chemicals", "specialty", "processing", "refining", "synthesis"},
     0.45, "Energy/Chemicals ↔ Pharma", 2),
]

# Terms too generic to justify a cross-domain bridge on their own. At least one
# bridge hit must come from OUTSIDE this set, so "precision + chemical" (both
# generic) no longer fires a Medical/Pharma bridge.
GENERIC_BRIDGE_TERMS = {
    "precision", "chemical", "chemicals", "assembly", "component", "components",
    "control", "device", "devices", "data", "software", "specialty", "processing",
}

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
    # generic filler that leaks into evidence but does not describe capability
    "used", "units", "local", "centers", "main", "line", "various", "range",
    "wide", "based", "provider", "private", "enterprise",
}
MIN_EVIDENCE_IDF = 1.5
# Evidence must DISCRIMINATE: a term appearing in more than this share of the
# corpus (e.g. "manufacturing", "industrial", "maintenance") is true of almost
# every company and is excluded from evidence even though it is a protected
# sector token and survives the common-word suppression.
EVIDENCE_MAX_DOC_RATIO = 0.22

# ------------------------------- text utils ------------------------------

# Corporate-trivia PHRASES stripped from company text before embedding. Phrase-
# level (not sentence-level) so a capability noun in the same sentence survives
# (e.g. "founded in 1990 and produces sulfuric acid" keeps "produces sulfuric
# acid"). Verified: capability text is preserved across all sectors.
_JUNK_PHRASES = re.compile("|".join([
    r"\bwas founded by [^.,;]+", r"\bfounded by [^.,;]+",
    r"\b(founded|established|incorporated)\s+(in\s+)?(18|19|20)\d{2}\b",
    r"\b(in|since)\s+(18|19|20)\d{2}\b", r"\b(18|19|20)\d{2}\b",
    r"\bis\s+headquartered\s+in\s+[^.,;]+", r"\bheadquartered\s+in\s+[^.,;]+",
    r"\bis\s+(a\s+)?(global,?\s+)?publicly[- ]traded\b", r"\bpublicly[- ]traded\b",
    r"\blisted on\s+[^.,;]+", r"\b(nasdaq|nyse)\b", r"\bstock exchange\b",
    r"\bengages in the provision of\b", r"\bwas incorporated\b",
]), re.I)
# Opportunity noise: prices/tonnages/percentages and geography/supplier names,
# which say nothing about capability fit.
_PRICE_NUM = re.compile(r"\b\d[\d,\.]*\s*(usd|ton|tons|%|percent|billion|million|sar|kg)?\b", re.I)
_GEO_NAMES = re.compile(
    r"\b(saudi|arabia|riyadh|jeddah|dammam|kingdom|mena|gcc|gulf|middle east|"
    r"asia[- ]?pacific|europe|africa|americas?|neom|sabic|tasnee|maaden|aramco|olayan)\b", re.I)


def focus_company_text(text) -> str:
    """Strip corporate/financial boilerplate phrases, keeping capability text."""
    return re.sub(r"\s+", " ", _JUNK_PHRASES.sub(" ", str(text or ""))).strip()


def focus_opportunity_text(text) -> str:
    """Strip prices, tonnages and geography from opportunity text."""
    return re.sub(r"\s+", " ", _GEO_NAMES.sub(" ", _PRICE_NUM.sub(" ", str(text or "")))).strip()


def preprocess(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Legal-form suffixes ignored when deciding whether two company names are the
# same entity ("Tuwaiq Casting & forging" == "Tuwaiq Casting and Forging",
# "Al Gurg ... LLC" == "Al Gurg ...").
_LEGAL_SUFFIXES = {"llc", "ltd", "limited", "inc", "co", "company", "gmbh",
                   "sa", "plc", "corp", "corporation"}


def canonical_name(name) -> str:
    """Canonical entity key for a company name (casefold, &->and, punctuation
    stripped, trailing legal-form suffixes removed)."""
    s = str(name or "").lower().replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    toks = [t for t in s.split() if t]
    while toks and toks[-1] in _LEGAL_SUFFIXES:
        toks.pop()
    return " ".join(toks)


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
        self.doc_ratio: dict = {}
        self.domain: set = set()

    def fit(self, corpus: list, sector_texts: list, extra_domain_texts: list):
        self.protected = set()
        for val in sector_texts:
            self.protected |= tokenize_static(val)
        for key, vals in SECTOR_ONTOLOGY.items():
            self.protected |= {key} | set(vals)
        for _, _, terms, _, _, _ in BRIDGE_RULES:
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
        self.doc_ratio = {tok: cnt / n for tok, cnt in doc_counts.items()}

        self.domain = set(self.protected)
        for text in extra_domain_texts:
            self.domain |= tokenize_static(text)
        self.domain = {t for t in self.domain if len(t) >= 4}

    def tokenize(self, text) -> set:
        return {t for t in tokenize_static(text) if t not in self.common}


VOCAB = Vocabulary()

# ------------------------------- embeddings -------------------------------


def _truthy(v) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def resolve_backends(args) -> dict:
    """Resolve the chat and embeddings backends independently.

    Two providers are supported: the public OpenAI API and Azure OpenAI
    (same setup as the uhnwi-fastapi project — MISA_USE_AZURE_OPENAI plus
    AZURE_OPENAI_ENDPOINT / _API_KEY / _API_VERSION / _DEPLOYMENT). Chat and
    embeddings are resolved separately because an Azure resource may host a
    chat deployment but no embeddings deployment (as merketfit.openai.azure.com
    does): in that case GPT validation runs on Azure while semantic vectors
    fall back to TF-IDF. On Azure the `model=` argument is a *deployment* name,
    not a model family.

    Returns a dict with: chat_client, chat_kind, chat_models (list),
    embed_client, embed_kind, embed_model.
    """
    out = dict(chat_client=None, chat_kind=None, chat_models=[],
               embed_client=None, embed_kind=None, embed_model=None)
    if args.no_openai:
        return out

    az_client = None
    if _truthy(os.getenv("MISA_USE_AZURE_OPENAI")):
        endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
        key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
        if endpoint and key:
            from openai import AzureOpenAI
            az_client = AzureOpenAI(
                azure_endpoint=endpoint, api_key=key,
                api_version=(os.getenv("AZURE_OPENAI_API_VERSION")
                             or AZURE_API_VERSION_DEFAULT).strip(),
                timeout=OPENAI_TIMEOUT, max_retries=OPENAI_RETRIES,
            )
            az_chat = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "").strip()
            az_embed = (os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
                        or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or "").strip()
            # --chat-provider public forces the gate onto the public-API model;
            # 'azure'/'auto' use the in-tenant deployment when present.
            if az_chat and getattr(args, "chat_provider", "auto") != "public":
                out.update(chat_client=az_client, chat_kind="azure", chat_models=[az_chat])
            if az_embed:
                out.update(embed_client=az_client, embed_kind="azure", embed_model=az_embed)

    pub_client = None
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        pub_client = OpenAI(timeout=OPENAI_TIMEOUT, max_retries=OPENAI_RETRIES)
    # Public API fills whichever backend Azure did not provide. With
    # --chat-provider azure we keep chat in-tenant only (no public fallback).
    want_public_chat = getattr(args, "chat_provider", "auto") != "azure"
    if out["chat_client"] is None and pub_client is not None and want_public_chat:
        out.update(chat_client=pub_client, chat_kind="openai", chat_models=list(GPT_MODELS))
    if out["embed_client"] is None and pub_client is not None:
        out.update(embed_client=pub_client, embed_kind="openai", embed_model=EMBEDDING_MODEL)
    return out


def _hash(text: str, model: str) -> str:
    return hashlib.md5(f"{model}::{text}".encode()).hexdigest()


def _load_emb_cache() -> dict:
    if os.path.exists(EMB_CACHE):
        loaded = np.load(EMB_CACHE)
        return {k: loaded[k] for k in loaded.files}
    return {}


def embed_texts(texts: list, client, model: str, cache: dict) -> np.ndarray:
    """Batched embeddings against a shared in-memory cache keyed by (model,
    text), so Azure and public-API vectors never collide. The caller owns
    loading/saving the cache once per run (embed_texts only fills it)."""
    missing = [t for t in texts if _hash(t, model) not in cache]
    for start in range(0, len(missing), EMBED_BATCH):
        chunk = missing[start:start + EMBED_BATCH]
        resp = client.embeddings.create(model=model, input=chunk)
        for text, item in zip(chunk, resp.data):
            cache[_hash(text, model)] = np.asarray(item.embedding, dtype=np.float32)
    return np.vstack([cache[_hash(t, model)] for t in texts])


def build_vectors(companies: pd.DataFrame, opps: pd.DataFrame, args, backends: dict):
    """Returns (profile_mat, product_mat, opp_mat, mode)."""
    corpus = (
        companies["combined"].tolist()
        + companies["products_clean"].tolist()
        + opps["requirement"].tolist()
    )
    client = backends["embed_client"]
    model = backends["embed_model"]
    kind = backends["embed_kind"]
    if client is not None and model:
        try:
            client.embeddings.create(model=model, input=["ping"])  # auth / deployment check
        except Exception as e:
            msg = (f"{kind} embeddings UNAVAILABLE ({type(e).__name__}): "
                   f"falling back to TF-IDF.")
            if args.require_openai:
                sys.exit(f"FATAL: {msg} (--require-openai set)")
            print(f"\n{'!' * 70}\n{msg}\nScores are NOT comparable with embedding-mode runs.\n{'!' * 70}\n")
            client = None

    if client is None and args.require_openai:
        sys.exit("FATAL: no embeddings backend available and --require-openai given "
                 "(Azure resources need a text-embedding deployment; see docs).")

    if client is not None:
        cache = _load_emb_cache()  # load once; embed_texts fills it in place
        n_before = len(cache)
        prof = embed_texts(companies["combined"].tolist(), client, model, cache)
        prod = embed_texts(companies["products_clean"].tolist(), client, model, cache)
        opp = embed_texts(opps["requirement"].tolist(), client, model, cache)
        # Capability-focused blend: sharpen the profile and opportunity vectors
        # with a boilerplate-stripped version. Blending keeps the ranking stable
        # (see EMBED_FOCUS_BLEND). Products are already capability-only, left as-is.
        w = getattr(args, "embed_blend", EMBED_FOCUS_BLEND)
        if w > 0 and "combined_focused" in companies.columns:
            def _unit(v):
                return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
            prof_f = embed_texts(companies["combined_focused"].tolist(), client, model, cache)
            opp_f = embed_texts(opps["requirement_focused"].tolist(), client, model, cache)
            prof = (1 - w) * _unit(prof) + w * _unit(prof_f)
            opp = (1 - w) * _unit(opp) + w * _unit(opp_f)
            kind = f"{kind}+focus{w:g}"
        if len(cache) > n_before:  # save once, only if new vectors were added
            os.makedirs(os.path.dirname(EMB_CACHE), exist_ok=True)
            np.savez_compressed(EMB_CACHE, **cache)
        return prof, prod, opp, kind

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
            vec(opps["requirement"]), "tfidf")


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
        for side_a, side_b, terms, bscore, name, min_hits in BRIDGE_RULES:
            crossed = (c & side_a and o & side_b) or (c & side_b and o & side_a)
            hits = sorted(capability_tokens & terms)
            # A bridge needs >= min_hits distinct capability terms AND at least one
            # that is not purely generic industrial vocabulary — otherwise a couple
            # of boilerplate words ("precision", "chemical") spuriously bridge an
            # industrial company into pharma/medtech (all such matches were rejected
            # by GPT).
            specific_hits = [h for h in hits if h not in GENERIC_BRIDGE_TERMS]
            if crossed and len(hits) >= min_hits and specific_hits and bscore > score:
                score = bscore
                bridge_name = name
                reason = f"Cross-sector bridge ({name}) via capabilities: {', '.join(hits[:6])}."
                break

    label = ("Strong" if score >= 0.80 else "Moderate" if score >= 0.50
             else "Weak" if score > 0 else "No")
    return score, label, reason, bridge_name

# --------------------------- evidence + fusion ----------------------------


def domain_overlap(company_tokens: set, opp_tokens: set, strict: bool = False) -> list:
    """Domain-vocabulary terms shared by both sides, ranked by IDF.

    Takes precomputed token sets (each company/opportunity is tokenized once,
    not once per pair). Geographic/market/filler words are always excluded.

    strict=True additionally drops corpus-frequent terms (a word shared by nearly
    every company, e.g. "manufacturing"). Strict terms are used for the shown
    evidence and the evidence score, so explanations DISCRIMINATE. The lenient
    list (strict=False) is used only to COUNT evidence for qualification, so
    tightening the displayed evidence never silently prunes a real candidate.
    """
    shared = company_tokens & opp_tokens & VOCAB.domain
    out = []
    for t in shared:
        if t in NON_CAPABILITY_TERMS or VOCAB.idf.get(t, 0.0) < MIN_EVIDENCE_IDF:
            continue
        if strict and VOCAB.doc_ratio.get(t, 0.0) > EVIDENCE_MAX_DOC_RATIO:
            continue
        out.append(t)
    return sorted(out, key=lambda t: VOCAB.idf.get(t, 0.0), reverse=True)


def evidence_score(terms: list) -> float:
    """Absolute 0-1 evidence strength from IDF mass of shared domain terms."""
    total = sum(VOCAB.idf.get(t, 0.0) for t in terms[:10])
    return float(1.0 - np.exp(-total / 8.0))


def decision_label(final: float, qualified: bool, real_sector_link: bool,
                   gpt_decision=None, bridge_only: bool = False) -> str:
    """Business label. When the GPT gate has graded a pair, its verdict governs:
      - "Direct"  → the company can itself execute the opportunity → High/Good Fit
      - "Partial" → a credible supplier/partner/adjacent player     → Partner Fit
      - "No"      → only superficial overlap                        → Low Fit
    ("Yes" is accepted as a legacy alias for Direct.)

    Without a GPT verdict the score-based ladder applies: High/Good Fit require a
    real sector relationship (exact/family/bridge); a bridge-only link caps at
    "Good Fit"; softer links cap at "Review Needed" so an analyst confirms.
    """
    if not qualified or gpt_decision in ("No", "Not Run"):
        return "Low Fit"
    if gpt_decision in ("Direct", "Yes"):
        return "High Fit" if final >= 0.60 else "Good Fit"
    if gpt_decision == "Partial":
        return "Partner Fit"
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


GPT_SYSTEM_PROMPT = (
    "You are a rigorous industrial investment-matching analyst. For a company and "
    "an investment opportunity you assess FIT as a graded verdict, drawing only on "
    "the evidence given — never invent capabilities. You distinguish a company that "
    "can directly execute the opportunity from one that is a credible partner or "
    "supplier to it, from one with only a superficial keyword/sector overlap."
)

# Ordered most- to least-conservative for tie-breaking (a tie resolves DOWN).
GPT_FIT_ORDER = ["No", "Partial", "Direct"]


# The rubric is a versioned artifact: RUBRIC_HASH is stamped on every label so
# calibration and the verdict-diff report can tell which judging regime a
# verdict came from. Any edit to GPT_SYSTEM_PROMPT or RUBRIC_CORE changes the
# hash automatically.
RUBRIC_CORE = """Grade how well this COMPANY fits this investment OPPORTUNITY, using
these tiers:

- "Direct": the company is in the opportunity's sector AND its stated products/
  services cover the core technical work (it could itself manufacture/assemble/
  deliver the opportunity's product). Name the matching capability.
- "Partial": the company does not make the finished product, but has a real,
  named linkage worth engaging — e.g. it supplies key components, materials, or
  technology the opportunity needs, or has strongly adjacent manufacturing that
  could credibly be extended. A concrete supplier/partner rationale is required.
- "None": only a generic sector or keyword overlap ("chemical", "precision",
  "assembly"), a different end-product, or no real linkage.

Be strict about the Direct/Partial boundary and the Partial/None boundary: a
plausible-sounding adjacency with no named component, material, or technology
linkage is "None". Commodity or generic-infrastructure supply (bulk chemicals,
standard filtration, generic machining) to a specialized opportunity is "None"
unless the opportunity's core work depends on that specific input — but when
such a genuine-yet-insufficient supplier link exists, NAME IT in the
explanation as context rather than ignoring it."""

EXEMPLAR_HEADER = (
    "ANALYST PRECEDENTS — verdicts on OTHER pairs from human review. They show\n"
    "the analyst's standards for THOSE specific pairings. Do NOT generalize a\n"
    "rejection beyond its case: judge THIS pair strictly on its own brief, and a\n"
    "company supplying an input the brief explicitly requires remains at least\n"
    "Partial regardless of the precedents below."
)

RUBRIC_HASH = hashlib.md5(
    (GPT_SYSTEM_PROMPT + RUBRIC_CORE + EXEMPLAR_HEADER).encode()
).hexdigest()[:8]


def build_exemplar_lines(human: dict, companies: pd.DataFrame) -> list:
    """Compact analyst-precedent lines for the gate prompt, one per human
    verdict: [(pair, line)]. Phrasing is deliberately PAIR-SPECIFIC — an early
    version said "this type of linkage" and the gate over-generalized, rejecting
    pairs the analyst had explicitly approved. The caller excludes the pair
    under judgment so the gate stays an independent judge of it."""
    by_name = {r["company_name"]: r for _, r in companies.iterrows()}
    out = []
    for (comp, opp), label in human.items():
        c = by_name.get(comp)
        prods = str(c["product and Services"])[:140] if c is not None else ""
        verdict = ("FIT — the analyst confirmed this supplier linkage for this opportunity"
                   if label == 1 else
                   "NOT A FIT — the analyst judged this particular linkage too peripheral for this opportunity")
        out.append(((comp, opp),
                    f"- {comp} -> {opp}: analyst verdict {verdict}. (Company offers: {prods})"))
    return out


def _gpt_prompt(comp: pd.Series, opp: pd.Series, exemplars: str = "") -> str:
    precedent = ""
    if exemplars:
        precedent = f"\n{EXEMPLAR_HEADER}\n{exemplars}\n"
    return f"""{RUBRIC_CORE}
{precedent}
Return STRICT JSON only:
{{"fit": "Direct|Partial|None", "confidence": 0.0-1.0,
  "explanation": "2-4 sentences naming the specific capability/linkage that fits, or the specific gap (and any weaker supplier link worth noting)"}}

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
  Demand Drivers: {opp['What are the key demand drivers?']}
  Required Materials: {opp['What materials are involved or required in the project?']}
"""


def _parse_fit(raw) -> str:
    s = str(raw).strip().lower()
    if s.startswith("direct") or s == "yes":
        return "Direct"
    if s.startswith("partial") or s.startswith("supplier") or s.startswith("partner"):
        return "Partial"
    return "No"


def _gpt_call_once(client, model: str, prompt: str):
    """One gate sample. Returns (fit, confidence, explanation) or raises."""
    resp = client.chat.completions.create(
        model=model, temperature=GPT_TEMPERATURE,
        messages=[
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    content = re.sub(r"^```(?:json)?|```$", "", content).strip()
    parsed = json.loads(content)
    fit = _parse_fit(parsed.get("fit", parsed.get("decision", "None")))
    conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    return fit, conf, str(parsed.get("explanation", "")).strip()


ESCALATION_VOTES = 2  # extra samples drawn when the first round is split


def gpt_validate(client, models: list, comp: pd.Series, opp: pd.Series,
                 votes: int = GPT_VOTES, escalate: bool = True, exemplars: str = ""):
    """Self-consistency gate: sample the first working model `votes` times and
    majority-vote a graded verdict (Direct / Partial / No). Returns
    (fit, confidence, explanation, model, agreement); confidence = (winning
    share) x (mean stated confidence of the winning tier).

    Escalation: a split first round (not unanimous) draws ESCALATION_VOTES more
    samples before tallying, so borderline pairs (the ones that used to flip
    between runs at 2/3) are decided on 5 votes instead of 3. Ties resolve to
    the more conservative tier so the gate never over-promises.
    """
    prompt = _gpt_prompt(comp, opp, exemplars=exemplars)
    for model in models:
        results = []
        for _ in range(max(1, votes)):
            try:
                results.append(_gpt_call_once(client, model, prompt))
            except Exception:
                break  # this model is unusable — fall through to the next
        if not results:
            continue
        if escalate and len(set(f for f, _, _ in results)) > 1:
            for _ in range(ESCALATION_VOTES):
                try:
                    results.append(_gpt_call_once(client, model, prompt))
                except Exception:
                    break
        n = len(results)
        counts = {tier: sum(1 for f, _, _ in results if f == tier) for tier in GPT_FIT_ORDER}
        top = max(counts.values())
        # Winner = most-voted tier; ties broken toward the more conservative tier
        # (GPT_FIT_ORDER lists conservative-first).
        fit = next(t for t in GPT_FIT_ORDER if counts[t] == top)
        winning = [r for r in results if r[0] == fit]
        mean_conf = sum(c for _, c, _ in winning) / len(winning)
        conf = round((top / n) * mean_conf, 2)
        explanation = max(winning, key=lambda r: r[1])[2]
        return fit, conf, explanation, model, f"{top}/{n}"
    return "Not Run", 0.0, "GPT validation unavailable.", None, "0/0"

# ------------------------------- calibration -------------------------------


def _unanimous_share(df: pd.DataFrame) -> float:
    """Share of graded pairs whose votes were unanimous ('k/n' with k == n)."""
    graded = df["gpt_agreement"][df["gpt_agreement"].astype(str).str.contains("/")]
    if not len(graded):
        return 0.0
    unanimous = sum(1 for a in graded if a.split("/")[0] == a.split("/")[1])
    return round(unanimous / len(graded), 3)


def load_human_reviews(path: str = HUMAN_REVIEWS_CSV) -> dict:
    """Analyst verdicts exported from the review GUI: {(company, opportunity):
    1|0}. Agree-style verdicts map to 1, disagree-style to 0; anything else is
    ignored. Missing or unreadable file -> empty dict (the loop is optional)."""
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"WARNING: could not read {path} ({type(e).__name__}); ignoring it.")
        return {}
    out = {}
    for _, r in df.iterrows():
        v = str(r.get("verdict", "")).strip().lower()
        if v in ("agree", "yes", "fit", "1", "true"):
            label = 1
        elif v in ("disagree", "no", "not a fit", "notfit", "0", "false"):
            label = 0
        else:
            continue
        out[(str(r.get("company", "")).strip(),
             str(r.get("opportunity", "")).strip())] = label
    return out


def load_prior_verdicts(path: str = LABELS_JSONL) -> dict:
    """Latest recorded gate verdict per pair, from before this run's append."""
    prior = {}
    try:
        with open(path) as fh:
            for line in fh:
                d = json.loads(line)
                prior[(d["company"], d["opportunity"])] = d
    except FileNotFoundError:
        pass
    return prior


def compute_verdict_flips(df: pd.DataFrame, prior: dict) -> list:
    """Pairs whose gate verdict changed vs the previous recorded verdict.
    Nothing may change silently: these are printed and written to the
    Verdict_Changes sheet every run."""
    flips = []
    for r in df.itertuples():
        g = r.gpt_decision
        if g not in ("Direct", "Partial", "Yes", "No"):
            continue
        p = prior.get((r.company, r.opportunity))
        if p and p["decision"] != g:
            flips.append({
                "company": r.company, "opportunity": r.opportunity,
                "previous_verdict": p["decision"],
                "previous_ts": p.get("ts", ""),
                "rubric_changed": "yes" if p.get("rubric", "") != RUBRIC_HASH else "no",
                "new_verdict": g, "agreement": r.gpt_agreement,
                "human_verdict": getattr(r, "human_verdict", ""),
            })
    return flips


def apply_human_overrides(df: pd.DataFrame, human: dict) -> int:
    """Analyst verdicts outrank the gate in the OUTPUT, not just in calibration.

    Requires a `validated_fit` boolean column (gate-endorsed pairs). A human
    "disagree" pulls the pair out of the validated set and forces Low Fit; a
    human "agree" validates the pair (Partner Fit if the label was pessimistic).
    Returns the number of pairs actually changed. The gate's own verdict columns
    are left untouched — the override is visible via human_verdict.
    """
    changed = 0
    for idx, r in df.iterrows():
        hv = human.get((r["company"], r["opportunity"]))
        if hv is None:
            continue
        if hv == 0 and (r["validated_fit"] or r["ai_decision"] != "Low Fit"):
            df.at[idx, "validated_fit"] = False
            df.at[idx, "ai_decision"] = "Low Fit"
            changed += 1
        elif hv == 1 and not r["validated_fit"]:
            df.at[idx, "validated_fit"] = True
            if r["ai_decision"] in ("", "Low Fit", "Review Needed"):
                df.at[idx, "ai_decision"] = "Partner Fit"
            changed += 1
    return changed


def calibrate_probability(df: pd.DataFrame, human_reviews: dict | None = None) -> dict | None:
    """Learn P(validated fit | final_score) and add a calibrated
    `match_probability` column (audit M1).

    Label pool: the gate's accumulated verdicts in gpt_labels.jsonl (latest
    verdict per pair wins), OVERRIDDEN by any human review for the same pair —
    analyst judgment is gold, the gate's is silver — and human labels carry
    HUMAN_LABEL_WEIGHT in the fit. One monotone feature, so the reported AUC is
    exactly final_score's rank-discrimination on the label pool. Skipped
    (returns None) until the pool has at least 30 pairs with 8 per class.
    """
    pool: dict = {}
    try:
        with open(LABELS_JSONL) as fh:
            for line in fh:
                d = json.loads(line)
                y = 1 if d["decision"] in ("Direct", "Partial", "Yes") else 0
                # Labels from an older rubric were judged under a different
                # standard: keep them (signal) but at reduced weight.
                w = 1.0 if d.get("rubric", "") == RUBRIC_HASH else 0.6
                pool[(d["company"], d["opportunity"])] = (float(d["final_score"]), y, w)
    except FileNotFoundError:
        pass

    n_human = 0
    if human_reviews:
        cur_score = {(r.company, r.opportunity): float(r.final_score)
                     for r in df.itertuples()}
        for pair, label in human_reviews.items():
            score = pool.get(pair, (None,))[0]
            if score is None:
                score = cur_score.get(pair)
            if score is None:
                continue  # reviewed pair no longer exists in the data
            pool[pair] = (score, label, HUMAN_LABEL_WEIGHT)
            n_human += 1

    if not pool:
        return None
    scores, ys, ws = zip(*pool.values())
    n_pos = sum(ys)
    if len(pool) < 30 or n_pos < 8 or (len(pool) - n_pos) < 8:
        return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = np.array(scores).reshape(-1, 1)
    lr = LogisticRegression().fit(X, ys, sample_weight=ws)
    auc = roc_auc_score(ys, lr.predict_proba(X)[:, 1])
    df["match_probability"] = lr.predict_proba(
        df[["final_score"]].to_numpy()
    )[:, 1].round(3)
    return {"n_labels": len(pool), "n_positive": n_pos, "n_human": n_human,
            "auc": round(float(auc), 3)}


# ------------------------- consortium (needs) view -------------------------
#
# Decomposes an opportunity into the capabilities it EXPLICITLY requires and
# maps gate-validated companies onto each need ("consortium view"). Guarded so
# thin briefs never produce invented value chains:
#   1. Quote-grounded extraction — every need must carry a verbatim quote from
#      the brief, and the quote is verified in code. No verified quote, no need.
#   2. Readiness gate — fewer than MIN_VERIFIED_NEEDS verified needs marks the
#      opportunity "thin data" and it keeps the plain ranked view (fallback).
# Needs with no covering validated company are reported as GAP, not hidden.


def _norm_quote(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", str(s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def quote_in_text(quote: str, text: str) -> bool:
    """True when the (normalized) quote genuinely appears in the source text
    and is substantial enough to ground a need (4+ words)."""
    q = _norm_quote(quote)
    return len(q.split()) >= 4 and q in _norm_quote(text)


def _needs_source_text(opp: pd.Series) -> str:
    fields = [
        "What is the opportunity description?",
        "What are the investment highlights?",
        "What is the value proposition of this opportunity?",
        "What materials are involved or required in the project?",
    ]
    return "\n".join(str(opp.get(f, "") or "") for f in fields)


def extract_needs(client, models: list, opp: pd.Series, cache: dict) -> list:
    """Extract quote-grounded needs for one opportunity. Returns only the needs
    whose quotes verify against the brief. Cached by text hash."""
    src = _needs_source_text(opp)
    key = hashlib.md5(f"needs::{src}".encode()).hexdigest()
    if key not in cache:
        prompt = f"""From the OPPORTUNITY text below, list the distinct technical capabilities,
production stages, or input supplies it EXPLICITLY requires.

Rules:
- Only include what the text explicitly states. Do not infer or invent.
- Each item: {{"capability": "<2-5 word label>", "quote": "<verbatim quote from
  the text, 5-25 words, copied exactly>"}}
- 3 to 10 items; return fewer (even zero) if the text lacks explicit detail.

Return STRICT JSON only: {{"needs": [...]}}

OPPORTUNITY TEXT:
{src}
"""
        parsed = {"needs": []}
        for model in models:
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=0,
                    messages=[{"role": "system", "content":
                               "You extract explicitly stated requirements from text. "
                               "You never infer beyond what is written."},
                              {"role": "user", "content": prompt}],
                )
                content = re.sub(r"^```(?:json)?|```$", "",
                                 (resp.choices[0].message.content or "").strip()).strip()
                parsed = json.loads(content)
                break
            except Exception:
                continue
        cache[key] = parsed
    needs = cache[key].get("needs", []) if isinstance(cache[key], dict) else []
    return [n for n in needs
            if isinstance(n, dict) and n.get("capability")
            and quote_in_text(n.get("quote", ""), src)]


def map_consortium(client, models: list, opp_name: str, needs: list,
                   validated: pd.DataFrame, comp_by: dict) -> list:
    """Map validated companies onto each verified need. Companies outside the
    validated set are dropped in post-processing, so nothing can be invented."""
    allowed = list(validated["company"])
    lines = []
    for name in allowed:
        c = comp_by[name]
        prods = str(c["product and Services"])[:400]
        lines.append(f"- {name} [{c['Sector']}]: {prods}")
    needs_txt = "\n".join(f"- {n['capability']} (from brief: \"{n['quote']}\")" for n in needs)
    prompt = f"""OPPORTUNITY: {opp_name}

VERIFIED NEEDS (each grounded in the brief):
{needs_txt}

VALIDATED COMPANIES (the only companies you may reference, names exactly as given):
{chr(10).join(lines)}

For each need, list which validated companies (zero or more) can credibly supply
or deliver it, each with a short reason grounded in their stated products.
Return STRICT JSON only:
{{"mapping": [{{"need": "...", "companies": [{{"name": "...", "why": "..."}}]}}]}}
"""
    for model in models:
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "system", "content":
                           "You map suppliers to requirements using only the companies provided."},
                          {"role": "user", "content": prompt}],
            )
            content = re.sub(r"^```(?:json)?|```$", "",
                             (resp.choices[0].message.content or "").strip()).strip()
            mapping = json.loads(content).get("mapping", [])
            for entry in mapping:
                entry["companies"] = [c for c in entry.get("companies", [])
                                      if isinstance(c, dict) and c.get("name") in allowed]
            return mapping
        except Exception:
            continue
    return []


def build_consortium(chat_client, chat_models, df: pd.DataFrame,
                     opps: pd.DataFrame, comp_by: dict, workers: int) -> tuple:
    """Returns (rows for the Consortium_View sheet, number of ready opps)."""
    cache = {}
    if os.path.exists(NEEDS_CACHE):
        try:
            cache = json.load(open(NEEDS_CACHE))
        except Exception:
            cache = {}
    # Respect analyst overrides: only pairs still validated after human review
    # may appear as need-coverers.
    if "validated_fit" in df.columns:
        validated_all = df[df["validated_fit"]]
    else:
        validated_all = df[df["gpt_decision"].isin(["Direct", "Partial", "Yes"])]

    def work(item):
        _, opp = item
        oname = opp["What is the opportunity name?"]
        needs = extract_needs(chat_client, chat_models, opp, cache)
        if len(needs) < MIN_VERIFIED_NEEDS:
            return [{"opportunity": oname,
                     "status": "Thin data - ranked view applies",
                     "need": "", "source_quote": "", "covered_by": "",
                     "why": "", "gap": ""}]
        vcomp = validated_all[validated_all["opportunity"] == oname]
        covered = {}
        if len(vcomp):
            for entry in map_consortium(chat_client, chat_models, oname, needs, vcomp, comp_by):
                covered[_norm_quote(entry.get("need", ""))] = entry.get("companies", [])
        def find_cover(capability):
            nc = _norm_quote(capability)
            for k, v in covered.items():
                if k and (nc == k or nc in k or k in nc):
                    return v
            return []

        rows = []
        for n in needs:
            comps = find_cover(n["capability"])
            rows.append({
                "opportunity": oname, "status": "Ready",
                "need": n["capability"], "source_quote": n["quote"],
                "covered_by": ", ".join(c["name"] for c in comps),
                "why": " | ".join(f"{c['name']}: {c.get('why', '')}" for c in comps),
                "gap": "" if comps else "GAP",
            })
        return rows

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 6))) as ex:
        results = list(ex.map(work, list(opps.iterrows())))
    try:
        os.makedirs(os.path.dirname(NEEDS_CACHE), exist_ok=True)
        json.dump(cache, open(NEEDS_CACHE, "w"))
    except Exception:
        pass
    rows = [r for rs in results for r in rs]
    ready = len({r["opportunity"] for r in rows if r["status"] == "Ready"})
    return rows, ready


# --------------------------------- loaders --------------------------------


def load_companies(dedupe: bool = True) -> pd.DataFrame:
    df = pd.read_excel(DATA_COMPANIES)
    df = df.rename(columns={
        "Company Name": "company_name", "Company Profile": "company_profile",
        "Product/Services": "product and Services",
    })
    required = ["company_name", "company_profile", "product and Services", "Sector"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"companies.xlsx missing columns: {missing}")

    # Entity resolution: rows whose canonical name matches are the same company
    # entered twice (e.g. "Tuwaiq Casting & forging" / "Tuwaiq Casting and
    # Forging"). Duplicates split scores and occupy two rank slots, so keep the
    # row with the richest text and drop the rest, loudly.
    merged = []
    if dedupe:
        df["_canon"] = df["company_name"].map(canonical_name)
        df["_richness"] = (df["company_profile"].astype(str).str.len()
                           + df["product and Services"].astype(str).str.len())
        for key, grp in df.groupby("_canon"):
            if len(grp) > 1:
                keep = grp["_richness"].idxmax()
                dropped = grp.drop(index=keep)["company_name"].tolist()
                merged.append((grp.loc[keep, "company_name"], dropped))
        if merged:
            keep_idx = df.groupby("_canon")["_richness"].idxmax()
            df = df.loc[sorted(keep_idx)]
            for kept, dropped in merged:
                print(f"Entity resolution: kept '{kept}', merged duplicate(s): {dropped}")
        df = df.drop(columns=["_canon", "_richness"])
    df.attrs["merged_entities"] = merged
    raw_combined = (
        df[["company_name", "company_profile", "product and Services"]]
        .astype(str).agg(" ".join, axis=1)
    )
    df["combined"] = raw_combined.apply(preprocess)
    # Focus is applied to the raw text (its regexes need punctuation), then
    # preprocessed like the rest. Used only by the embedding blend.
    df["combined_focused"] = raw_combined.apply(lambda t: preprocess(focus_company_text(t)))
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
    raw_req = df.apply(lambda r: " ".join(str(r.get(f, "")) for f in fields), axis=1)
    df["requirement"] = raw_req.apply(preprocess)
    df["requirement_focused"] = raw_req.apply(lambda t: preprocess(focus_opportunity_text(t)))
    return df.reset_index(drop=True)

# ----------------------------------- main ----------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gpt", action="store_true", help="skip GPT validation")
    ap.add_argument("--no-openai", action="store_true", help="force TF-IDF fallback")
    ap.add_argument("--require-openai", action="store_true",
                    help="fail hard instead of falling back to TF-IDF")
    ap.add_argument("--top-n", type=int, default=GPT_TOP_N_PER_OPPORTUNITY)
    ap.add_argument("--gpt-votes", type=int, default=GPT_VOTES,
                    help="self-consistency samples per pair for the GPT gate")
    ap.add_argument("--workers", type=int, default=GPT_WORKERS,
                    help="concurrent GPT gate validations")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="skip company entity resolution (keep duplicate rows)")
    ap.add_argument("--no-escalate", action="store_true",
                    help="disable extra gate votes on split first rounds")
    ap.add_argument("--no-consortium", action="store_true",
                    help="skip the needs/consortium view")
    ap.add_argument("--human-reviews", default=HUMAN_REVIEWS_CSV,
                    help="CSV of analyst verdicts exported from the review GUI")
    ap.add_argument("--embed-blend", type=float, default=EMBED_FOCUS_BLEND,
                    help="weight (0-1) on the capability-focused embedding; "
                         "0 = disable (exact pre-focus behaviour)")
    ap.add_argument("--chat-provider", choices=["auto", "azure", "public"], default="auto",
                    help="which backend runs the GPT gate. 'public' uses the "
                         "public-API gpt-4.1 (best quality); 'azure' keeps chat "
                         "in-tenant (data residency); 'auto' prefers Azure when set.")
    ap.add_argument("--env-file", default=None,
                    help="extra .env to load (e.g. to reuse the uhnwi Azure "
                         "credentials); overrides values from the local .env")
    args = ap.parse_args()

    if load_dotenv is not None:
        load_dotenv(".env")  # local project .env, if present
        if args.env_file:
            load_dotenv(args.env_file, override=True)
    elif args.env_file:
        sys.exit("FATAL: --env-file given but python-dotenv is not installed "
                 "(pip install python-dotenv).")

    backends = resolve_backends(args)

    companies = load_companies(dedupe=not args.no_dedupe)
    n_merged = sum(len(d) for _, d in companies.attrs.get("merged_entities", []))
    opps = load_opportunities()
    print(f"Loaded {len(companies)} companies "
          f"({n_merged} duplicate rows merged), {len(opps)} opportunities.")

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

    prof_mat, prod_mat, opp_mat, mode = build_vectors(companies, opps, args, backends)
    chat_client = backends["chat_client"]
    chat_models = backends["chat_models"]
    print(f"Embedding backend: {mode.upper()}  |  "
          f"Chat backend: {(backends['chat_kind'] or 'none').upper()}"
          f"{' (' + chat_models[0] + ')' if chat_models else ''}")

    sim_profile = cosine_matrix(prof_mat, opp_mat)
    sim_product = cosine_matrix(prod_mat, opp_mat)
    pct_profile = percentile_rank(sim_profile)
    pct_product = percentile_rank(sim_product)

    # FIX 8 — specificity: how much a pair beats the company's own average
    spec_profile = percentile_rank(sim_profile - sim_profile.mean(axis=1, keepdims=True))
    spec_product = percentile_rank(sim_product - sim_product.mean(axis=1, keepdims=True))
    sem_profile = (1 - SPECIFICITY_BLEND) * pct_profile + SPECIFICITY_BLEND * spec_profile
    sem_product = (1 - SPECIFICITY_BLEND) * pct_product + SPECIFICITY_BLEND * spec_product

    # Tokenize each company and opportunity ONCE (not once per pair): at the real
    # 2,960 x 309 scale this avoids ~900k redundant tokenizations.
    comp_text = [f"{c['company_profile']} {c['product and Services']}"
                 for _, c in companies.iterrows()]
    comp_cap = [expand(canon(VOCAB.tokenize(t))) for t in comp_text]
    comp_ev = [VOCAB.tokenize(t) for t in comp_text]
    opp_ev = [VOCAB.tokenize(o["requirement"]) for _, o in opps.iterrows()]

    rows = []
    for i, comp in companies.iterrows():
        cap_tokens = comp_cap[i]
        for j, opp in opps.iterrows():
            s_score, s_label, s_reason, bridge = sector_score(
                comp["Sector"], opp["Sector"], cap_tokens
            )
            qual_terms = domain_overlap(comp_ev[i], opp_ev[j])          # count -> qualification
            terms = domain_overlap(comp_ev[i], opp_ev[j], strict=True)  # discriminating -> display/score
            display_terms = terms or qual_terms  # never blank a qualified row
            ev = evidence_score(terms)

            # Soft candidate: no sector overlap and no bridge, only semantic +
            # domain-evidence signal. Flagged for transparency but never qualified
            # (pure semantic similarity is too noisy to stand in for a real sector
            # relationship, and produced nonsense picks such as a transformer maker
            # topping an MRI opportunity). No score mutation: the flag is enough.
            soft_match = (s_label in ("No", "Unknown")
                          and len(qual_terms) >= MIN_EVIDENCE_TERMS
                          and max(sem_profile[i, j], sem_product[i, j]) >= SOFT_MATCH_MIN_PCT)

            real_sector_link = s_label in ("Exact", "Strong", "Moderate") or bridge is not None
            bridge_only = bridge is not None and s_label not in ("Exact", "Strong", "Moderate")
            qualified = (not soft_match and s_score >= MIN_SECTOR_SCORE
                         and len(qual_terms) >= MIN_EVIDENCE_TERMS)
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
                "evidence_terms": ", ".join(display_terms[:8]),
                "n_evidence_terms": len(qual_terms),
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

    # Human reviews load BEFORE the gate: they feed it as analyst precedents
    # (few-shot exemplars) and later override the output. Prior verdicts load
    # before this run appends, for the drift report.
    human = load_human_reviews(args.human_reviews)
    if human:
        print(f"Human reviews loaded: {len(human)} verdicts from {args.human_reviews}.")
    exemplar_pairs = build_exemplar_lines(human, companies)
    prior_verdicts = load_prior_verdicts()

    # FIX 9 — GPT validation on frozen scores, qualified top-N only
    df["gpt_decision"] = ""
    df["gpt_confidence"] = np.nan
    df["gpt_explanation"] = ""
    df["gpt_agreement"] = ""
    if not args.no_gpt and chat_client is not None:
        todo = df[(df["rank_for_opportunity"] <= args.top_n) & df["qualified"]]
        print(f"GPT-validating {len(todo)} qualified top-{args.top_n} pairs "
              f"via {backends['chat_kind'].upper()} ({chat_models[0]}), "
              f"{args.gpt_votes}-vote self-consistency, {args.workers} workers...")

        # gpt_validate is pure (no shared state), so run the pairs concurrently
        # and write results back to the frame afterwards (single-threaded writes).
        def _validate(item):
            idx, row = item
            # Analyst precedents, excluding the pair under judgment so the gate
            # stays an independent judge of it (disagreement stays visible).
            lines = [l for p, l in exemplar_pairs
                     if p != (row["company"], row["opportunity"])][-8:]
            return idx, row, gpt_validate(
                chat_client, chat_models, companies.loc[row["_i"]], opps.loc[row["_j"]],
                votes=args.gpt_votes, escalate=not args.no_escalate,
                exemplars="\n".join(lines),
            )

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            results = list(ex.map(_validate, list(todo.iterrows())))

        labels = []
        for idx, row, (decision, conf, expl, model, agreement) in results:
            df.at[idx, "gpt_decision"] = decision
            df.at[idx, "gpt_confidence"] = conf
            df.at[idx, "gpt_explanation"] = expl
            df.at[idx, "gpt_agreement"] = agreement
            df.at[idx, "ai_decision"] = decision_label(
                row["final_score"], row["qualified"], row["real_sector_link"],
                gpt_decision=decision, bridge_only=row["bridge_only"],
            )
            labels.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "company": row["company"], "opportunity": row["opportunity"],
                "decision": decision, "confidence": conf, "agreement": agreement,
                "model": model, "votes": args.gpt_votes, "rubric": RUBRIC_HASH,
                "final_score": row["final_score"], "embed_mode": mode,
            })
        if labels:
            os.makedirs(os.path.dirname(LABELS_JSONL), exist_ok=True)
            with open(LABELS_JSONL, "a") as fh:
                for item in labels:
                    fh.write(json.dumps(item) + "\n")
            print(f"Appended {len(labels)} verdicts to {LABELS_JSONL}")
    elif not args.no_gpt:
        print("GPT validation skipped: no chat backend available "
              "(set a public OPENAI_API_KEY or Azure chat deployment).")

    # When GPT ran, "Good Fit"/"High Fit" must be GPT-backed. Only the top-N per
    # opportunity are validated, so a deeper qualified pair (rank > N) keeps a
    # model-only optimistic label that GPT never confirmed — and given GPT
    # rejected every validated top-N here, those deeper pairs almost certainly
    # would be rejected too. Cap any GPT-unseen qualified pair at "Review Needed"
    # so no sheet claims a fit GPT hasn't confirmed. (gpt_decision == "" marks
    # unseen pairs; "Yes"/"No" pairs already carry a GPT-backed label.)
    gpt_ran = (not args.no_gpt) and (chat_client is not None)
    if gpt_ran:
        unseen_optimistic = (
            df["qualified"] & (df["gpt_decision"] == "")
            & df["ai_decision"].isin(["Good Fit", "High Fit"])
        )
        df.loc[unseen_optimistic, "ai_decision"] = "Review Needed"
        print(f"Capped {int(unseen_optimistic.sum())} GPT-unseen qualified pairs "
              f"from Good/High Fit to Review Needed.")

    df = df.drop(columns=["_i", "_j"])

    # Human-in-the-loop: analyst verdicts override the gate in the output
    # (validated set + labels), feed calibration at higher weight, and are
    # surfaced as their own column so overrides stay visible.
    df["validated_fit"] = df["gpt_decision"].isin(["Direct", "Partial", "Yes"])
    n_over = 0
    df["human_verdict"] = [
        {1: "Agree", 0: "Disagree"}.get(human.get((r.company, r.opportunity), -1), "")
        for r in df.itertuples()
    ]
    if human:
        n_over = apply_human_overrides(df, human)
        print(f"Analyst overrides applied to {n_over} pair(s).")

    # Drift report: no gate verdict may change silently between runs.
    flips = compute_verdict_flips(df, prior_verdicts)
    if flips:
        print(f"VERDICT CHANGES vs previous run ({len(flips)}):")
        for f in flips:
            tag = "rubric changed" if f["rubric_changed"] == "yes" else "same rubric"
            print(f"  {f['company'][:26]:28} -> {f['opportunity'][:40]:42} "
                  f"{f['previous_verdict']} -> {f['new_verdict']}  ({tag})")
    elif gpt_ran:
        print("Verdict drift: none vs previous run.")

    # Calibrated probability from the accumulated label pool (audit M1).
    cal = calibrate_probability(df, human_reviews=human)
    if cal:
        print(f"Calibration: match_probability fitted on {cal['n_labels']} labeled "
              f"pairs ({cal['n_positive']} positive, {cal['n_human']} human), "
              f"final_score AUC = {cal['auc']}.")
    else:
        print("Calibration skipped: label pool too small (needs 30+ pairs, 8 per class).")

    # FIX 7 + GPT-aware abstention — an opportunity abstains when it has no
    # qualified candidate at all, OR (when GPT ran) when the gate graded every
    # validated top-N candidate as "No" (no Direct executor and no Partial
    # partner/supplier). The bridges qualify industrial companies for pharma/
    # medtech opps, but if the gate finds no real linkage the opportunity has no
    # validated fit in the current company set and should say so.
    abstained = []
    for opp_name, grp in df.groupby("opportunity"):
        if not grp["qualified"].any():
            best = grp.sort_values("final_score", ascending=False).iloc[0]
            abstained.append({
                "opportunity": opp_name,
                "status": "No qualified candidate",
                "best_candidate": best["company"],
                "detail": "No company clears the sector + evidence bar.",
            })
            continue
        if gpt_ran:
            validated = grp[grp["gpt_decision"].isin(["Direct", "Partial", "Yes", "No"])]
            if len(validated) and not grp["validated_fit"].any():
                best = validated.sort_values("final_score", ascending=False).iloc[0]
                abstained.append({
                    "opportunity": opp_name,
                    "status": "No validated fit (no Direct or Partial)",
                    "best_candidate": best["company"],
                    "detail": (f"{int(len(validated))} top candidates graded, none a "
                               f"direct executor or credible partner; best was "
                               f"{best['company']} (gate: No)."),
                })
    abstain_df = pd.DataFrame(abstained)

    # Consortium view (readiness-gated). Thin briefs fall back to the ranked
    # view; rich briefs get a quote-grounded needs->suppliers map with GAPs.
    consortium_rows, consortium_ready = [], 0
    if gpt_ran and not args.no_consortium:
        comp_by = {r["company_name"]: r for _, r in companies.iterrows()}
        print("Building consortium view (quote-grounded needs, readiness-gated)...")
        consortium_rows, consortium_ready = build_consortium(
            chat_client, chat_models, df, opps, comp_by, args.workers)
        n_gaps = sum(1 for r in consortium_rows if r["gap"] == "GAP")
        print(f"Consortium: {consortium_ready}/{len(opps)} opportunities ready, "
              f"{len(opps) - consortium_ready} fell back (thin data), {n_gaps} capability gaps.")

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
        {"metric": "opportunities_with_direct_fit",
         "value": int(df[df["gpt_decision"].isin(["Direct", "Yes"])]["opportunity"].nunique()) if gpt_ran else "n/a (no GPT)"},
        {"metric": "opportunities_with_partner_fit",
         "value": int(df[df["gpt_decision"] == "Partial"]["opportunity"].nunique()) if gpt_ran else "n/a (no GPT)"},
        {"metric": "pairs_direct_fit", "value": int((df["gpt_decision"].isin(["Direct", "Yes"])).sum()) if gpt_ran else "n/a"},
        {"metric": "pairs_partner_fit", "value": int((df["gpt_decision"] == "Partial").sum()) if gpt_ran else "n/a"},
        {"metric": "median_raw_profile_cosine", "value": round(float(df["raw_profile_cosine"].median()), 4)},
        {"metric": "top3_company_concentration",
         "value": round(df[df["rank_for_opportunity"] <= 3]["company"].value_counts(normalize=True).max(), 3)},
        {"metric": "duplicate_rows_merged", "value": n_merged},
        {"metric": "gate_unanimous_share", "value": _unanimous_share(df) if gpt_ran else "n/a"},
        {"metric": "calibration_labels", "value": cal["n_labels"] if cal else "insufficient"},
        {"metric": "calibration_human_labels", "value": cal["n_human"] if cal else len(human)},
        {"metric": "human_overrides_applied", "value": n_over},
        {"metric": "rubric_version", "value": RUBRIC_HASH},
        {"metric": "gate_verdict_flips_vs_prev", "value": len(flips)},
        {"metric": "calibration_auc_final_score", "value": cal["auc"] if cal else "insufficient"},
        {"metric": "consortium_ready_opportunities",
         "value": consortium_ready if (gpt_ran and not args.no_consortium) else "n/a"},
    ])

    os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        opp_view.to_excel(writer, sheet_name="Opportunity_View", index=False)
        comp_view.to_excel(writer, sheet_name="Company_View", index=False)
        df.sort_values(["opportunity", "rank_for_opportunity"]).to_excel(
            writer, sheet_name="All_Pairs", index=False)
        (abstain_df if len(abstain_df) else pd.DataFrame([{"opportunity": "-", "status": "All opportunities have a validated candidate"}])).to_excel(
            writer, sheet_name="Abstentions", index=False)
        if consortium_rows:
            pd.DataFrame(consortium_rows).to_excel(
                writer, sheet_name="Consortium_View", index=False)
        if flips:
            pd.DataFrame(flips).to_excel(writer, sheet_name="Verdict_Changes", index=False)
        diag.to_excel(writer, sheet_name="Diagnostics", index=False)

    print(f"\nSaved {OUTPUT_XLSX}")
    print(diag.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
