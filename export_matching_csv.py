#!/usr/bin/env python3
"""Export matches in the MatchingOutput CSV schema.

Emits exactly these columns, matching the reference file
(Downloads/MatchingOutput vq.csv):

  id, companyId, opportunityId, company_sector, opportunity_sector,
  sector_similarity, profile_similarity, product_similarity, ai_score,
  ai_decision, final_score, ai_explanation, rank, ai_insight,
  suggested_plan, match_reason

Structure mirrors the reference: each company gets its top-N (default 5)
opportunities ranked 1..N, including the rejected ones (ai_decision = No).

Scores come from the last matching_v2 run (Output/matches_v2.xlsx). The four
narrative fields (ai_explanation, ai_insight, suggested_plan, match_reason) are
generated per pair by one GPT call, run in a thread pool.

Conventions taken from the reference file:
  - sector_similarity is 1 only when company_sector == opportunity_sector.
  - ai_score is 1 for Yes, 0 for No.
  - suggested_plan / match_reason are JSON arrays of 3 strings.
  - ai_decision maps from the graded gate: Direct or Partial -> Yes, None -> No.

IDs are sequential surrogates (companies 1..N in spreadsheet order, likewise
opportunities). They do NOT correspond to any production database.

Usage:
  python3 export_matching_csv.py --env-file "/path/to/.env"
  python3 export_matching_csv.py --top-n 5 --workers 8 --out Output/matching_output.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

MODEL_CHAIN = ["gpt-4.1", "gpt-4o", "gpt-4o-mini"]

COLUMNS = [
    "id", "companyId", "opportunityId", "company_sector", "opportunity_sector",
    "sector_similarity", "profile_similarity", "product_similarity", "ai_score",
    "ai_decision", "final_score", "ai_explanation", "rank", "ai_insight",
    "suggested_plan", "match_reason",
]

SYSTEM = (
    "You are a rigorous investment-matching analyst assessing whether a company "
    "fits a Saudi investment opportunity. Judge only from the evidence given; "
    "never invent capabilities. Be concrete and name specifics."
)

_print_lock = threading.Lock()


def prompt_for(comp, opp) -> str:
    return f"""Assess this COMPANY against this OPPORTUNITY and return strict JSON.

Grade "fit" as one of:
- "Direct": the company could itself manufacture/assemble/deliver the
  opportunity's product (its stated products cover the core work).
- "Partial": it cannot make the finished product, but has a real named linkage
  worth engaging (supplies key components, materials or technology, or has
  strongly adjacent manufacturing that could credibly extend).
- "None": only a generic sector or keyword overlap, a different end-product, or
  no real linkage.

Return STRICT JSON only, no markdown:
{{
  "fit": "Direct|Partial|None",
  "ai_explanation": "3-5 sentences assessing sector, profile and product fit. State plainly what the company can and cannot do for this opportunity.",
  "ai_insight": "2-3 sentences on the strategic implication: what this company unlocks for the opportunity or the Saudi market if engaged.",
  "suggested_plan": ["3 concrete actions to pitch or engage this company", "...", "..."],
  "match_reason": ["3 concise, specific reasons this pairing does or does not work", "...", "..."]
}}

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
  Value proposition: {opp['What is the value proposition of this opportunity?']}
  Demand drivers: {opp['What are the key demand drivers?']}
  Required materials: {opp['What materials are involved or required in the project?']}
"""


def _as3(v) -> list:
    """Coerce a model-returned list into exactly 3 strings."""
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        v = []
    out = [str(x).strip() for x in v if str(x).strip()]
    return (out + [""] * 3)[:3] if out else ["", "", ""]


def generate(client, comp, opp) -> dict:
    prompt = prompt_for(comp, opp)
    for model in MODEL_CHAIN:
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.2,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}],
            )
            content = (resp.choices[0].message.content or "").strip()
            content = re.sub(r"^```(?:json)?|```$", "", content).strip()
            p = json.loads(content)
            fit = str(p.get("fit", "None")).strip().lower()
            fit = ("Direct" if fit.startswith("direct")
                   else "Partial" if fit.startswith("partial") else "None")
            return {
                "fit": fit,
                "ai_explanation": str(p.get("ai_explanation", "")).strip(),
                "ai_insight": str(p.get("ai_insight", "")).strip(),
                "suggested_plan": _as3(p.get("suggested_plan")),
                "match_reason": _as3(p.get("match_reason")),
            }
        except Exception:
            continue
    return {"fit": "None", "ai_explanation": "", "ai_insight": "",
            "suggested_plan": ["", "", ""], "match_reason": ["", "", ""]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", default="Output/matches_v2.xlsx")
    ap.add_argument("--out", default="Output/matching_output.csv")
    ap.add_argument("--top-n", type=int, default=5, help="opportunities per company")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N rows")
    args = ap.parse_args()

    if load_dotenv is not None:
        load_dotenv(".env")
        if args.env_file:
            load_dotenv(args.env_file, override=True)

    if not os.path.exists(args.xlsx):
        raise SystemExit(f"Not found: {args.xlsx} (run matching_v2.py first).")

    from matching_v2 import load_companies, load_opportunities
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set (use --env-file).")
    client = OpenAI()

    companies = load_companies()
    opps = load_opportunities()
    comp_by = {r["company_name"]: r for _, r in companies.iterrows()}
    opp_by = {r["What is the opportunity name?"]: r for _, r in opps.iterrows()}
    company_id = {n: i + 1 for i, n in enumerate(companies["company_name"])}
    opportunity_id = {n: i + 1 for i, n in enumerate(opps["What is the opportunity name?"])}

    allp = pd.read_excel(args.xlsx, sheet_name="All_Pairs")
    # Rank every opportunity per company (rejected rows included, like the
    # reference file), then keep the top N.
    allp["rank"] = (allp.groupby("company")["final_score"]
                    .rank(method="first", ascending=False).astype(int))
    sel = allp[allp["rank"] <= args.top_n].sort_values(["company", "rank"]).reset_index(drop=True)
    if args.limit:
        sel = sel.head(args.limit)
    print(f"Generating narrative fields for {len(sel)} pairs "
          f"(top {args.top_n} per company) with {args.workers} workers...")

    done = [0]

    def work(i_row):
        i, row = i_row
        g = generate(client, comp_by[row["company"]], opp_by[row["opportunity"]])
        with _print_lock:
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(sel):
                print(f"  {done[0]}/{len(sel)}")
        return i, g

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = dict(ex.map(work, list(sel.iterrows())))

    rows = []
    for i, row in sel.iterrows():
        g = results[i]
        yes = g["fit"] in ("Direct", "Partial")
        cs, os_ = str(row["company_sector"]).strip(), str(row["opportunity_sector"]).strip()
        rows.append({
            "id": len(rows) + 1,
            "companyId": company_id[row["company"]],
            "opportunityId": opportunity_id[row["opportunity"]],
            "company_sector": cs,
            "opportunity_sector": os_,
            "sector_similarity": 1 if cs == os_ else 0,
            "profile_similarity": round(float(row["raw_profile_cosine"]), 3),
            "product_similarity": round(float(row["raw_product_cosine"]), 3),
            "ai_score": 1 if yes else 0,
            "ai_decision": "Yes" if yes else "No",
            "final_score": round(float(row["final_score"]), 3),
            "ai_explanation": g["ai_explanation"],
            "rank": int(row["rank"]),
            "ai_insight": g["ai_insight"],
            "suggested_plan": json.dumps(g["suggested_plan"], ensure_ascii=False),
            "match_reason": json.dumps(g["match_reason"], ensure_ascii=False),
        })

    out = pd.DataFrame(rows, columns=COLUMNS)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    n_yes = int((out["ai_decision"] == "Yes").sum())
    print(f"\nWrote {args.out}: {len(out)} rows, {n_yes} Yes / {len(out) - n_yes} No, "
          f"{out['companyId'].nunique()} companies x top-{args.top_n}.")


if __name__ == "__main__":
    main()
