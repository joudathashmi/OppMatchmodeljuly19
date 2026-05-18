"""
Multi-signal triage helpers for business_grade_matching (keyword overlap + STEM).
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Sequence, Set

try:
    from nltk.stem import PorterStemmer
except ImportError:
    PorterStemmer = None  # type: ignore


_stemmer = PorterStemmer() if PorterStemmer else None

# LLM / shorthand labels → exact KPMG Sector string (applied only if that string is in allowed).
_COMMON_ADJACENT_ALIASES: dict[str, str] = {
    "pharmaceutical": "Pharma & Biotech",
    "pharmaceuticals": "Pharma & Biotech",
    "pharma": "Pharma & Biotech",
    "biopharma": "Pharma & Biotech",
    "biotech": "Pharma & Biotech",
    "life sciences": "Healthcare and Life Sciences",
    "healthcare": "Healthcare and Life Sciences",
    "medical devices": "Medical Equipment Manufacturing",
    "medical device": "Medical Equipment Manufacturing",
    "chemicals": "Chemical Manufacturing",
    "oil and gas": "Oil and Gas",
    "oil, gas, energy & water": "Oil, Gas, Energy & Water",
    "renewable energy": "Renewable Energy Semiconductor Manufacturing",
    "ict": "Information and Communication Technology",
    "information technology": "Information and Communication Technology",
}


def parse_json_list(cell: Any) -> List[str]:
    """Parse Excel JSON-string list column."""
    if cell is None:
        return []
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    s = str(cell).strip()
    if not s:
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return []


def resolve_adjacent_label(raw: Any, allowed: Set[str]) -> Optional[str]:
    """Map a single adjacent-sector phrase to an exact string in allowed, or None."""
    s = str(raw).strip() if raw is not None else ""
    if not s or not allowed:
        return None
    if s in allowed:
        return s
    low = s.lower()
    ci = [a for a in allowed if a.lower() == low]
    if len(ci) == 1:
        return ci[0]
    mapped = _COMMON_ADJACENT_ALIASES.get(low)
    if mapped and mapped in allowed:
        return mapped
    # Unambiguous substring (avoid short tokens like "oil" matching many rows).
    if len(low) < 6:
        return None
    sub = [a for a in allowed if low in a.lower() or a.lower() in low]
    if len(sub) == 1:
        return sub[0]
    return None


def resolve_adjacent_labels(items: Sequence[str], allowed: Set[str]) -> List[str]:
    """Deduped list of allowed sectors, order preserved."""
    out: List[str] = []
    seen: Set[str] = set()
    for x in items:
        r = resolve_adjacent_label(x, allowed)
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _corp_stems(corpus: str) -> set:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{1,}", (corpus or "").lower())
    if _stemmer is None:
        return set(words)
    return {_stemmer.stem(w) for w in words if len(w) > 2}


def keyword_match_count(phrases: Sequence[str], corpus: str) -> int:
    """Case-insensitive substring hit OR all stemmed tokens of phrase found in corpus."""
    corp_l = (corpus or "").lower()
    stems = _corp_stems(corpus)
    hits = 0
    for ph in phrases:
        raw = str(ph).strip()
        if len(raw) < 2:
            continue
        rl = raw.lower()
        if rl and rl in corp_l:
            hits += 1
            continue
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{1,}", raw.lower())
        if _stemmer is None:
            if tokens and all(t in corp_l for t in tokens if len(t) > 2):
                hits += 1
            continue
        stoks = {_stemmer.stem(t) for t in tokens if len(t) > 2}
        if stoks and stoks <= stems:
            hits += 1
    return hits


def format_pass_reason(parts: List[str]) -> str:
    return " + ".join(parts) if parts else "unknown"


def format_fail_reason(
    *,
    overlap: int,
    adj: bool,
    ck: int,
    inp: int,
    sig_p: float,
    sig_pf: float,
) -> str:
    return (
        "no_pass"
        f" overlap={overlap}, adj={int(adj)}, capability_kw={ck}, inputs={inp}, "
        f"signal_product_sim={sig_p:.3f}, signal_profile_sim={sig_pf:.3f}"
    )
