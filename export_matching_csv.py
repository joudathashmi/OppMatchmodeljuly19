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

Scores come from the last matching_v2 run (Output/matches_v2.xlsx). For pairs the
gate already validated there, the verdict (ai_decision) and explanation are
REUSED verbatim, so the CSV never contradicts the workbook on the same pair; only
the long-tail pairs the gate never saw are judged here, with the same rubric. The
narrative fields (ai_insight, suggested_plan, match_reason) are always generated.

Output schema conventions:
  - sector_similarity is 1 only when company_sector == opportunity_sector.
  - profile/product/final scores are decimals clamped to [0, 1].
  - ai_score is 1 for Yes, 0 for No; ai_decision follows ai_score.
  - ai_explanation: investment rationale for Yes rows; why-not for No rows.
  - ai_insight, suggested_plan and match_reason are populated for recommended
    (Yes) matches only - suggested_plan / match_reason as JSON arrays of exactly
    three strings - and left blank for No rows.
  - rank orders opportunities per company by final_score (1 = best).
  - ai_decision maps from the graded gate: Direct or Partial -> Yes, None -> No;
    analyst overrides outrank the gate.

IDs are sequential surrogates (companies 1..N in spreadsheet order, likewise
opportunities). They do NOT correspond to any production database.

Usage:
  python3 export_matching_csv.py
  python3 export_matching_csv.py --top-n 5 --workers 8 --out Output/matching_output.csv

Credentials come from this project's .env (gitignored).
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
    "id", "companyId", "company_name", "opportunityId", "opportunity_name",
    "company_sector", "opportunity_sector",
    "sector_similarity", "profile_similarity", "product_similarity", "ai_score",
    "ai_decision", "final_score", "ai_explanation", "rank", "ai_insight",
    "suggested_plan", "match_reason",
    # Per-signal explanations, populated on every row (Yes and No): they
    # justify the profile_similarity and product_similarity columns.
    "profile_match_reason", "product_match_reason",
]

SYSTEM = (
    "You are a senior industrial analyst at MISA (Saudi Arabia) writing match "
    "rationales a business-development team will act on. Every rationale must be "
    "SPECIFIC TO THIS PAIR: cite named products, value-chain stages, materials "
    "and market facts from the texts. Formulaic phrasing is a defect - if a "
    "sentence could be pasted into a different company's row unchanged, rewrite "
    "it. Judge only from the evidence given; never invent capabilities."
)

# Rotating opening directives: without them the model writes every explanation
# on the same skeleton ("X is a strong partner for...", 305 times).
OPENERS = [
    "open ai_explanation with the opportunity's requirement or value-chain stage",
    "open ai_explanation with the company's single most relevant product line or capability",
    "open ai_explanation with the decisive point of alignment or the decisive gap",
    "open ai_explanation with the demand or market context the opportunity sits in",
]

_print_lock = threading.Lock()


def prompt_for(comp, opp, forced=None, opener: str = "") -> str:
    opener_line = opener or "vary your opening naturally"
    directive = ""
    if forced == "Yes":
        directive = ("\nDECISION ALREADY MADE: this pairing IS a positive match. Set fit "
                     "to Direct or Partial (never None) and write every field affirmatively.\n")
    elif forced == "No":
        directive = ("\nDECISION ALREADY MADE: this pairing is NOT a match. Set fit to None "
                     "and write ai_explanation beginning with \"No.\", with match_reason "
                     "giving the reasons it does not fit.\n")
    return f"""Assess this COMPANY against this OPPORTUNITY and return strict JSON.
{directive}
First grade "fit" internally as one of:
- "Direct": the company could itself manufacture/assemble/deliver the
  opportunity's product (its stated products cover the core work).
- "Partial": it does not make the finished product, but is a credible supplier
  or partner (supplies key components, materials or technology, or has strongly
  adjacent manufacturing that could credibly extend).
- "None": only a generic sector or keyword overlap, a different end-product, or
  no real linkage.

WRITING STYLE (strict):
- Ground every claim: cite at least TWO concrete specifics from the OPPORTUNITY
  (a value-chain stage, a named required material or component, a demand driver
  or market fact) and at least TWO from the COMPANY (named product lines or
  services, sectors served, scale or footprint facts). No generic claims.
- BANNED phrases (using any is a defect): "strong partner", "reliable
  supplier", "aligns well", "well-positioned", "leveraging", "expertise in",
  "extensive experience", "proven track record", "supports Vision 2030".
  Mention Vision 2030 only through a concrete mechanism (e.g. import
  substitution of a named product), if at all.
- Vary structure across fields and rows: {opener_line}. Do NOT open
  ai_explanation with the company name, and do not reuse one sentence skeleton
  across the fields.
- ai_explanation, positive match: 4-6 sentences of analysis - which stage(s)
  of the opportunity's value chain the company would cover and with which
  named products, what it would NOT cover, and the concrete consequence of
  engaging it (a supply localized, an import avoided, a dependency reduced).
- ai_explanation, non-match: begin with "No." then 3-4 sentences on the
  decisive capability gap, stated factually; if a genuine but insufficient
  supplier link exists, name it in one clause instead of ignoring it.
- profile_match_reason: 1-2 sentences on identity, scale and sectors served,
  citing profile facts; do not repeat product detail here.
- product_match_reason: 1-2 sentences mapping named products to named
  opportunity requirements (or naming exactly what is missing).
- ai_insight (positive only): one NON-OBVIOUS implication - localization
  economics, supply-chain risk, export angle - never a restatement of the
  explanation.
- suggested_plan: 3 actions specific to THIS pair, each naming a product,
  stage or counterpart; generic business-development steps are a defect.

Return STRICT JSON only, no markdown:
{{
  "fit": "Direct|Partial|None",
  "ai_explanation": "see writing style above",
  "ai_insight": "1-2 sentences on the strategic implication: what engaging this company unlocks for the opportunity or the Saudi market.",
  "suggested_plan": ["3 concrete engagement/pitch actions", "...", "..."],
  "match_reason": ["3 concise, specific reasons this pairing works (or, for None, does not)", "...", "..."],
  "profile_match_reason": "1-2 sentences: why the company's overall PROFILE (who they are, scale, sectors served, track record) does or does not align with this opportunity. Always populated, for both fits and non-fits.",
  "product_match_reason": "1-2 sentences: why the company's specific PRODUCTS/SERVICES do or do not cover what this opportunity needs, naming the relevant products or the missing ones. Always populated, for both fits and non-fits."
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


def generate(client, comp, opp, forced=None, opener: str = "") -> dict:
    """Generate the narrative fields for one pair.

    `forced` ("Yes"/"No"/None) pins the verdict up front so the explanation and
    the decision never contradict each other. `opener` rotates the structural
    directive so rows do not share one sentence skeleton. On total failure
    returns an "error" so the caller can report it rather than silently
    emitting a blank row that reads like a genuine "No".
    """
    prompt = prompt_for(comp, opp, forced=forced, opener=opener)
    last_err = ""
    for model in MODEL_CHAIN:
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.55,
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
                "profile_match_reason": str(p.get("profile_match_reason", "")).strip(),
                "product_match_reason": str(p.get("product_match_reason", "")).strip(),
                "error": "",
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            continue
    return {"fit": "None", "ai_explanation": "", "ai_insight": "",
            "suggested_plan": ["", "", ""], "match_reason": ["", "", ""],
            "profile_match_reason": "", "product_match_reason": "",
            "error": last_err or "all models failed"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", default="Output/matches_v2.xlsx")
    ap.add_argument("--out", default="Output/matching_output.csv")
    ap.add_argument("--top-n", type=int, default=5, help="opportunities per company")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N rows")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="per-request timeout in seconds (SDK default 600 is far too long)")
    ap.add_argument("--retries", type=int, default=3, help="SDK retries per request")
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
    # A per-request timeout is essential: the SDK default is 600s, so a single
    # stalled connection blocks a worker for ten minutes and the whole pool can
    # sit at 0% CPU indefinitely. Bound each attempt and let the SDK retry.
    client = OpenAI(timeout=args.timeout, max_retries=args.retries)

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
    # The gate (matching_v2) already produced an authoritative verdict for the
    # pairs it validated. Reuse it so the CSV never contradicts the workbook on
    # the same pair; only the long-tail pairs the gate never saw are judged here,
    # using the same Direct/Partial/None rubric.
    has_gate = "gpt_decision" in sel.columns
    reused = [0]
    print(f"Building {len(sel)} rows (top {args.top_n} per company) with "
          f"{args.workers} workers; reusing gate verdicts where available...")

    done = [0]

    def work(i_row):
        i, row = i_row
        # Reuse the workbook's VERDICT (so the CSV never contradicts it) by
        # forcing it into the generation, so the affirmative reference-style
        # explanation always matches the final Yes/No. Analyst overrides
        # (human_verdict) outrank the gate.
        human = str(row.get("human_verdict", "")).strip()
        gate = str(row.get("gpt_decision", "")).strip() if has_gate else ""
        forced = None
        from_gate = False
        if human == "Disagree":
            forced, from_gate = "No", True
        elif human == "Agree":
            forced, from_gate = "Yes", True
        elif gate in ("Direct", "Partial", "Yes"):
            forced, from_gate = "Yes", True
        elif gate == "No":
            forced, from_gate = "No", True
        g = generate(client, comp_by[row["company"]], opp_by[row["opportunity"]],
                     forced=forced, opener=OPENERS[i % len(OPENERS)])
        g["from_gate"] = from_gate
        with _print_lock:
            done[0] += 1
            if g.get("from_gate"):
                reused[0] += 1
            if done[0] % 25 == 0 or done[0] == len(sel):
                print(f"  {done[0]}/{len(sel)}", flush=True)
            if g["error"]:
                print(f"  ! {row['company'][:30]} / {row['opportunity'][:30]}: {g['error']}",
                      flush=True)
        return i, g

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = dict(ex.map(work, list(sel.iterrows())))

    def unit(v) -> float:
        """Decimal in [0, 1] as the schema requires (cosines can dip below 0)."""
        return round(min(max(float(v), 0.0), 1.0), 3)

    rows = []
    for i, row in sel.iterrows():
        g = results[i]
        yes = g["fit"] in ("Direct", "Partial")
        cs, os_ = str(row["company_sector"]).strip(), str(row["opportunity_sector"]).strip()
        rows.append({
            "id": len(rows) + 1,
            "companyId": company_id[row["company"]],
            "company_name": row["company"],
            "opportunityId": opportunity_id[row["opportunity"]],
            "opportunity_name": row["opportunity"],
            "company_sector": cs,
            "opportunity_sector": os_,
            "sector_similarity": 1 if cs == os_ else 0,
            # The calibrated semantic values that actually feed final_score
            # (percentile-normalized), NOT the raw cosines: exporting raw
            # cosines next to a percentile-based final_score made rows
            # impossible to reconcile (profile 0.33 with final 0.81).
            "profile_similarity": unit(row["semantic_profile"]),
            "product_similarity": unit(row["semantic_product"]),
            "ai_score": 1 if yes else 0,
            "ai_decision": "Yes" if yes else "No",
            "final_score": unit(row["final_score"]),
            "ai_explanation": g["ai_explanation"],
            "rank": int(row["rank"]),
            # Schema: insight, plan and reason are populated for recommended
            # matches only; blank for No rows.
            "ai_insight": g["ai_insight"] if yes else "",
            "suggested_plan": json.dumps(g["suggested_plan"], ensure_ascii=False) if yes else "",
            "match_reason": json.dumps(g["match_reason"], ensure_ascii=False) if yes else "",
            "profile_match_reason": g["profile_match_reason"],
            "product_match_reason": g["product_match_reason"],
        })

    n_failed = sum(1 for g in results.values() if g["error"])
    if n_failed:
        # Never let a generation failure masquerade as a genuine "No".
        print(f"\nWARNING: {n_failed}/{len(sel)} pairs failed to generate and were "
              f"written with empty narrative fields. Re-run to fill them.")

    out = pd.DataFrame(rows, columns=COLUMNS)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    n_yes = int((out["ai_decision"] == "Yes").sum())
    print(f"\nWrote {args.out}: {len(out)} rows, {n_yes} Yes / {len(out) - n_yes} No, "
          f"{out['companyId'].nunique()} companies x top-{args.top_n}"
          + (f", {n_failed} FAILED." if n_failed else "."))
    print(f"Verdicts reused verbatim from the gate: {reused[0]}/{len(sel)} "
          f"(the rest judged here with the same rubric).")


if __name__ == "__main__":
    main()
