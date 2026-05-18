"""
Infer a missing company Sector using a closed vocabulary (from non-blank sectors
in the same workbook), optional web snippets, and a single JSON LLM call per company.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Sequence, Set

import pandas as pd
from tqdm import tqdm

WEB_TIMEOUT_SEC = 10.0
EVIDENCE_MAX = 200
UNKNOWN = "UNKNOWN"

SECTOR_INFER_SYSTEM = f"""You map a company to exactly one sector from CLOSED_VOCABULARY, or {UNKNOWN} if none are a reasonable fit.

Rules:
- sector must equal one string from CLOSED_VOCABULARY exactly, or {UNKNOWN}.
- source_tag must be one of: row_only, web_only, row_and_web (whether row fields vs web snippets drove the decision).
- evidence: one short factual phrase (≤200 characters) citing row text or web; no fabricated URLs.

Return strict JSON with keys: sector, source_tag, evidence.
"""


def _safe_str(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v).strip()


def clip_evidence(s: str, n: int = EVIDENCE_MAX) -> str:
    s = _safe_str(s)
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def closed_sector_vocabulary(companies: pd.DataFrame, sector_col: str = "Sector") -> List[str]:
    seen: Set[str] = set()
    for v in companies[sector_col].astype(str).map(_safe_str):
        if v:
            seen.add(v)
    return sorted(seen)


def _cache_key(company_id: int, company_name: str) -> str:
    n = " ".join(_safe_str(company_name).lower().split())
    return f"id:{int(company_id)}:{n}"


def load_sector_inference_cache(path: str) -> Dict[str, Any]:
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_sector_inference_cache(path: str, cache: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _strip_html(html_fragment: str) -> str:
    t = re.sub(r"<[^>]+>", " ", html_fragment)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def fetch_web_snippets(
    query: str,
    *,
    timeout: float = WEB_TIMEOUT_SEC,
    max_snippets: int = 5,
) -> List[str]:
    q = _safe_str(query)
    if not q:
        return []
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(q)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) sector-infer/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []
    chunks = re.findall(
        r'class="(?:result__snippet|web-result__description)[^"]*"[^>]*>(.*?)</',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    out: List[str] = []
    for raw in chunks:
        txt = _strip_html(raw)
        if len(txt) < 24:
            continue
        out.append(txt[:900])
        if len(out) >= max_snippets:
            break
    return out


def _row_bundle(row: pd.Series) -> Dict[str, str]:
    return {
        "company_name": _safe_str(row.get("company_name", "")),
        "company_profile": _safe_str(row.get("company_profile", ""))[:2400],
        "product_services": _safe_str(row.get("product_services", ""))[:2400],
        "industry_field": _safe_str(row.get("industry_field", "")),
        "business_unit_field": _safe_str(row.get("business_unit_field", "")),
        "hq_field": _safe_str(row.get("hq_field", "")),
        "country_field": _safe_str(row.get("country_field", "")),
        "website_field": _safe_str(row.get("website_field", "")),
    }


def _coerce_llm_pick(
    sector: str,
    source_tag: str,
    evidence: str,
    vocabulary: Sequence[str],
) -> tuple[str, str, str]:
    vocab_set = set(vocabulary)
    s = _safe_str(sector)
    if s not in vocab_set and s != UNKNOWN:
        s = UNKNOWN
    allowed_tags = {"row_only", "web_only", "row_and_web"}
    tag = _safe_str(source_tag).lower().replace("-", "_")
    if tag not in allowed_tags:
        tag = "row_only"
    return s, tag, clip_evidence(evidence)


def infer_one_company(
    client: Any,
    call_json_fn: Callable[..., dict],
    model: str,
    vocabulary: Sequence[str],
    row: pd.Series,
    web_snippets: List[str],
) -> Dict[str, Any]:
    bundle = _row_bundle(row)
    payload = {
        "CLOSED_VOCABULARY": list(vocabulary),
        "company_row": bundle,
        "web_snippets": web_snippets,
    }
    parsed = call_json_fn(client, model, SECTOR_INFER_SYSTEM, payload)
    sector, tag, ev = _coerce_llm_pick(
        str(parsed.get("sector", "")),
        str(parsed.get("source_tag", "")),
        str(parsed.get("evidence", "")),
        vocabulary,
    )
    return {
        "sector": sector if sector else UNKNOWN,
        "source_tag": tag,
        "evidence": ev,
    }


def enrich_companies_missing_sectors(
    client: Any,
    companies: pd.DataFrame,
    *,
    vocabulary: Sequence[str],
    cache_path: str,
    call_json_fn: Callable[..., dict],
    model: str,
) -> pd.DataFrame:
    """Mutates ``companies`` in place (adds _effective_sector and audit columns)."""
    df = companies
    vocab_list = list(vocabulary)
    if not vocab_list:
        print("  [sector inference] No closed vocabulary — skipping inference.")
        df["_effective_sector"] = df["Sector"].astype(str).map(_safe_str)
        df["sector_was_inferred"] = False
        df["sector_inference_source"] = "none"
        df["sector_inference_evidence"] = ""
        return df

    cache = load_sector_inference_cache(cache_path)
    mutated = False

    df["_effective_sector"] = df["Sector"].astype(str).map(_safe_str)
    df["sector_was_inferred"] = False
    df["sector_inference_source"] = "none"
    df["sector_inference_evidence"] = ""

    need_mask = df["_effective_sector"].eq("")
    idxs = list(df.index[need_mask])
    if not idxs:
        print("  [sector inference] All companies have non-blank Sector — nothing to infer.")
        return df

    print(f"  [sector inference] Closed vocabulary size: {len(vocab_list)}; "
          f"companies needing sector: {len(idxs)}")
    for ix in tqdm(idxs, desc="  Sector infer (cache + web + LLM)"):
        row = df.loc[ix]
        cid = int(row["_company_id"])
        name = _safe_str(row.get("company_name", ""))
        key = _cache_key(cid, name)

        if key in cache:
            ent = cache[key]
            sec = _safe_str(ent.get("sector")) or UNKNOWN
            if sec not in set(vocab_list) and sec != UNKNOWN:
                sec = UNKNOWN
            df.at[ix, "_effective_sector"] = sec
            df.at[ix, "sector_was_inferred"] = True
            df.at[ix, "sector_inference_source"] = _safe_str(ent.get("source_tag")) or "row_only"
            df.at[ix, "sector_inference_evidence"] = clip_evidence(ent.get("evidence", ""))
            continue

        bundle = _row_bundle(row)
        name_q = bundle["company_name"] or bundle["website_field"]
        blob = (
            bundle["company_profile"][:280]
            + " "
            + bundle["product_services"][:280]
            + " "
            + bundle["industry_field"]
        )
        query = f"{name_q} {_safe_str(blob)[:220]}".strip()

        snippets = fetch_web_snippets(query) if query else []
        # If LLM receives no web text, encourage source_tag consistency
        try:
            result = infer_one_company(
                client, call_json_fn, model, vocab_list, row, snippets,
            )
        except Exception as e:
            result = {"sector": UNKNOWN, "source_tag": "row_only", "evidence": f"llm_error:{type(e).__name__}"}

        df.at[ix, "_effective_sector"] = result["sector"]
        df.at[ix, "sector_was_inferred"] = True
        df.at[ix, "sector_inference_source"] = result["source_tag"]
        df.at[ix, "sector_inference_evidence"] = result["evidence"]

        cache[key] = {
            "sector": result["sector"],
            "source_tag": result["source_tag"],
            "evidence": result["evidence"],
        }
        mutated = True

    if mutated:
        save_sector_inference_cache(cache_path, cache)
    return df
