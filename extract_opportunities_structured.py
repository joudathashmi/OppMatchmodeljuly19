#!/usr/bin/env python3
"""
Phase 1 — Build opportunities_structured.xlsx from new_opportunities.xlsx.

One LLM JSON call extracts five structured field groups per opportunity.
adjacent_value_chain_sectors is constrained to exact strings from kpmgfile Sector column.

Usage:
  python extract_opportunities_structured.py [--output PATH] [--source PATH]

Requires OPENAI_API_KEY (or .env next to this script).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Set

import pandas as pd
from openai import OpenAI

# Reuse matcher helpers (same working directory)
from business_grade_matching import (
    COMPANY_FILE_CANDIDATES,
    call_json,
    concat_matching_excel_sheets,
    find_first_existing,
    load_project_dotenv,
    normalize_company_cols,
    safe_str,
    to_list,
    ONTOLOGY_MODEL,
)
from matcher_signals import resolve_adjacent_labels
from sector_inference import closed_sector_vocabulary

OPP_SOURCE_DEFAULT = ["new_opportunities.xlsx", "Data/new_opportunities.xlsx"]
OUTPUT_DEFAULT = "opportunities_structured.xlsx"

STRUCT_SYSTEM = """You are an industrial investment analyst structuring opportunity briefs into machine-readable fields.

Rules:
1. Fill all five structured lists — be specific, grounded in the opportunity text only.
2. adjacent_value_chain_sectors: each entry MUST appear verbatim in ALLOWED_SECTORS. No synonyms or new labels. If unsure, omit rather than guessing.
3. precedent_industries: free-form industry labels (regions/markets analogy).
4. capability_keywords and required_capabilities/inputs must be concise phrases useful for substring and keyword retrieval.

Return strict JSON matching the requested schema."""

STRUCT_SCHEMA_DOC = {
    "results": [
        {
            "opportunity_index": "integer  // 0-based row index in INPUT_OPPORTUNITIES order",
            "required_capabilities": ["string"],
            "required_inputs": ["string"],
            "adjacent_value_chain_sectors": ["string  // MUST be verbatim from ALLOWED_SECTORS"],
            "precedent_industries": ["string"],
            "capability_keywords": ["string"],
        },
    ]
}


def _strip_header(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\xa0", " ").replace("\u200b", "").strip() for c in df.columns]
    return df


def load_raw_opportunities(path: str) -> pd.DataFrame:
    """Single-sheet or multi-sheet; stack like matcher but without strict rename yet."""
    book = pd.read_excel(path, sheet_name=None)
    chunks: List[pd.DataFrame] = []
    for name, raw in book.items():
        if raw is None or getattr(raw, "empty", True):
            continue
        chunks.append(_strip_header(raw))
    if not chunks:
        raise ValueError(f"No rows in {path!r}")
    return pd.concat(chunks, ignore_index=True)


def row_to_narrative_bundle(i: int, row: pd.Series) -> dict:
    return {
        "opportunity_index": i,
        "opportunity_name": safe_str(row.get("What is the opportunity name?", "")),
        "sector": safe_str(row.get("Sector", "")),
        "opportunity_description": safe_str(row.get("What is the opportunity description?", "")),
        "investment_highlights": safe_str(row.get("What are the investment highlights?", "")),
        "value_proposition": safe_str(row.get("What is the value proposition of this opportunity?", "")),
        "demand_drivers": safe_str(row.get("What are the key demand drivers?", "")),
        "key_players": safe_str(row.get("Who are the key players in this sector or project?", "")),
        "materials_required": safe_str(row.get("What materials are involved or required in the project?", "")),
        "market_data": safe_str(row.get("Market data", "")),
        "cost_structure": safe_str(row.get("Cost structure", "")),
        "government_incentives": safe_str(row.get("Government incentives", "")),
        "risks_and_mitigations": safe_str(row.get("Risks and mitigations", "")),
        "investment_locations": safe_str(row.get("Investment locations", "")),
    }


def filter_adjacent_to_vocab(items: List[str], allowed: Set[str]) -> List[str]:
    return resolve_adjacent_labels([safe_str(x) for x in items if safe_str(x)], allowed)


def main() -> None:
    load_project_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="", help="Opportunity workbook (default: first of OPP_SOURCE_DEFAULT)")
    ap.add_argument("--output", default=OUTPUT_DEFAULT, help=f"Default: {OUTPUT_DEFAULT}")
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY is required.", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    company_file = find_first_existing(COMPANY_FILE_CANDIDATES)
    if os.path.basename(company_file) != "kpmgfile.xlsx":
        print("Warning: company file is not kpmgfile.xlsx; vocabulary may not match deployment.", file=sys.stderr)

    companies = concat_matching_excel_sheets(company_file, normalize_company_cols, "Companies")
    allowed_sectors: Set[str] = set(closed_sector_vocabulary(companies))
    allowed_list = sorted(allowed_sectors)

    opp_path = args.source or find_first_existing(OPP_SOURCE_DEFAULT)
    raw = load_raw_opportunities(opp_path)
    required = [
        "What is the opportunity name?",
        "What is the opportunity description?",
        "Sector",
    ]
    miss = [c for c in required if c not in raw.columns]
    if miss:
        raise SystemExit(f"Opportunity file missing columns: {miss}")

    bundles = [row_to_narrative_bundle(i, raw.iloc[i]) for i in range(len(raw))]

    payload = {
        "ALLOWED_SECTORS": allowed_list,
        "INPUT_OPPORTUNITIES": bundles,
        "instructions": (
            f"Produce exactly {len(bundles)} objects in results[], one per opportunity_index 0..{len(bundles)-1}. "
            "Every opportunity_index must appear once."
        ),
        "output_schema": STRUCT_SCHEMA_DOC,
    }
    print(f"Calling {ONTOLOGY_MODEL} for structured extraction ({len(bundles)} opportunities)…")
    parsed = call_json(client, ONTOLOGY_MODEL, STRUCT_SYSTEM, payload)
    by_idx: Dict[int, dict] = {}
    for r in parsed.get("results", []):
        try:
            idx = int(r.get("opportunity_index", -1))
        except (TypeError, ValueError):
            continue
        by_idx[idx] = r

    missing_idx = [i for i in range(len(bundles)) if i not in by_idx]
    if missing_idx:
        raise SystemExit(f"LLM omitted opportunity_index rows: {missing_idx}")

    # Build structured columns
    caps: List[str] = []
    inputs: List[str] = []
    adjacents: List[str] = []
    precedents: List[str] = []
    keywords: List[str] = []
    warns: List[str] = []

    for i in range(len(bundles)):
        r = by_idx[i]
        rc = to_list(r.get("required_capabilities"))
        ri = to_list(r.get("required_inputs"))
        adj_raw = to_list(r.get("adjacent_value_chain_sectors"))
        adj = filter_adjacent_to_vocab(adj_raw, allowed_sectors)
        if len(adj) < len(adj_raw):
            dropped = set(safe_str(x) for x in adj_raw) - set(adj)
            warns.append(f"Row {i}: dropped non-vocab adjacent sectors {dropped!r}")
        pr = to_list(r.get("precedent_industries"))
        ck = to_list(r.get("capability_keywords"))

        caps.append(json.dumps(rc, ensure_ascii=False))
        inputs.append(json.dumps(ri, ensure_ascii=False))
        adjacents.append(json.dumps(adj, ensure_ascii=False))
        precedents.append(json.dumps(pr, ensure_ascii=False))
        keywords.append(json.dumps(ck, ensure_ascii=False))

    for w in warns[:20]:
        print(f"  [warn] {w}")
    if len(warns) > 20:
        print(f"  … and {len(warns) - 20} more warnings")

    # Output frame: snake_case narrative + structured JSON-string columns
    out = pd.DataFrame(
        {
            "opportunity_name": raw["What is the opportunity name?"].map(lambda x: safe_str(x)),
            "sector": raw["Sector"].map(lambda x: safe_str(x)),
            "opportunity_description": raw["What is the opportunity description?"].map(lambda x: safe_str(x)),
            "investment_highlights": raw["What are the investment highlights?"].fillna("").map(safe_str),
            "value_proposition": raw["What is the value proposition of this opportunity?"].fillna("").map(safe_str),
            "demand_drivers": raw["What are the key demand drivers?"].fillna("").map(safe_str),
            "key_players": raw["Who are the key players in this sector or project?"].fillna("").map(safe_str),
            "materials_required": raw["What materials are involved or required in the project?"].fillna("").map(safe_str),
            "market_data": raw["Market data"].fillna("").map(safe_str),
            "cost_structure": raw["Cost structure"].fillna("").map(safe_str),
            "government_incentives": raw["Government incentives"].fillna("").map(safe_str),
            "risks_and_mitigations": raw["Risks and mitigations"].fillna("").map(safe_str),
            "investment_locations": raw["Investment locations"].fillna("").map(safe_str),
            "required_capabilities": caps,
            "required_inputs": inputs,
            "adjacent_value_chain_sectors": adjacents,
            "precedent_industries": precedents,
            "capability_keywords": keywords,
        },
    )

    out_path = args.output
    out.to_excel(out_path, index=False)
    print(f"\nWrote {len(out)} rows → {out_path!r}")

    print("\n=== Per-opportunity structured fields (human spot-check) ===")
    for i, r in out.iterrows():
        print(f"\n--- [{i}] {r['opportunity_name'][:120]!r} ---")
        print("  required_capabilities:", r["required_capabilities"][:400])
        print("  required_inputs:", r["required_inputs"][:400])
        print("  adjacent_value_chain_sectors:", r["adjacent_value_chain_sectors"][:400])
        print("  precedent_industries:", r["precedent_industries"][:400])
        print("  capability_keywords:", r["capability_keywords"][:400])


if __name__ == "__main__":
    main()
