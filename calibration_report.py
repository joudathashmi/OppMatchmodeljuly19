#!/usr/bin/env python3
"""
Phase 4 — Compare two match exports for calibration / regression (no tuning).

Example:
  python3 calibration_report.py \\
    --baseline Output/business_grade_company_opportunity_matches_baseline.xlsx \\
    --new Output/business_grade_company_opportunity_matches.xlsx

Optional two-sheet workbook (regressed baseline-"Yes" rows + ``triage_reason`` counts):

  python3 calibration_report.py --baseline … --new … \\
    --write-regression-xlsx Output/calibration_regressed_baseline_yes_pairs.xlsx
"""

from __future__ import annotations

import argparse
from typing import FrozenSet, Tuple

import pandas as pd

Pair = Tuple[int, int]


def regression_dropped_keys(b: pd.DataFrame, n: pd.DataFrame, merge_key: list[str]) -> FrozenSet[Pair]:
    """Keys that were Yes in baseline and are not Yes in the new export."""
    by = b[b["ai_decision"] == "Yes"][merge_key].drop_duplicates()
    ny = n[n["ai_decision"] == "Yes"][merge_key].drop_duplicates()
    baseline_yes = frozenset(map(tuple, by.itertuples(index=False, name=None)))
    new_yes = frozenset(map(tuple, ny.itertuples(index=False, name=None)))
    return frozenset(baseline_yes - new_yes)


def write_regression_workbook(
    b: pd.DataFrame,
    n: pd.DataFrame,
    merge_key: list[str],
    path: str,
) -> int:
    """Sheet ``regressed_pairs`` plus ``by_triage_reason`` summary. Returns row count."""
    dropped = regression_dropped_keys(b, n, merge_key)
    if not dropped:
        empty_detail = pd.DataFrame(columns=list(merge_key))
        empty_summary = pd.DataFrame(columns=["triage_reason", "count", "mean_final_score"])
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            empty_detail.to_excel(writer, sheet_name="regressed_pairs", index=False)
            empty_summary.to_excel(writer, sheet_name="by_triage_reason", index=False)
        return 0

    keys_df = pd.DataFrame(list(dropped), columns=list(merge_key))
    reg_new = keys_df.merge(n, on=merge_key, how="left")
    reg_base = keys_df.merge(b, on=merge_key, how="left")
    dup_cols = [c for c in reg_base.columns if c not in merge_key]
    reg_base = reg_base.rename(columns={c: f"{c}_baseline" for c in dup_cols})

    detail = reg_new.merge(reg_base, on=merge_key, how="left")

    if "triage_reason" in detail.columns:
        g = detail.groupby("triage_reason", dropna=False)
        summary = g.size().reset_index(name="count")
        if "final_score" in detail.columns:
            means = g["final_score"].mean().reset_index(name="mean_final_score")
            summary = summary.merge(means, on="triage_reason", how="left")
        summary = summary.sort_values("count", ascending=False)
    else:
        summary = pd.DataFrame({"note": ["no triage_reason column on new export"]})

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        detail.to_excel(writer, sheet_name="regressed_pairs", index=False)
        summary.to_excel(writer, sheet_name="by_triage_reason", index=False)

    return len(detail)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration / regression between two match exports.")
    ap.add_argument("--baseline", required=True, help="Previous export workbook (.xlsx)")
    ap.add_argument("--new", required=True, help="New pipeline export workbook (.xlsx)")
    ap.add_argument("--top", type=int, default=30, help="Rows to show for new Yes / old Filtered")
    ap.add_argument(
        "--write-regression-xlsx",
        default="",
        help="Optional path: write regressing baseline-Yes pairs + triage_reason summary (two sheets)",
    )
    args = ap.parse_args()

    b = pd.read_excel(args.baseline, engine="openpyxl")
    n = pd.read_excel(args.new, engine="openpyxl")
    merge_key = ["companyId", "opportunityId"]

    print("TOTAL PAIRS new:", len(n))
    print("\n--- new ai_decision ---")
    print(n["ai_decision"].value_counts().to_string())

    merged = n.merge(b[merge_key + ["ai_decision"]], on=merge_key, suffixes=("_new", "_base"), how="left")

    baseline_yes_keys = set(
        tuple(x) for x in b[b["ai_decision"] == "Yes"][merge_key].itertuples(index=False, name=None)
    )
    new_yes_keys = set(
        tuple(x) for x in n[n["ai_decision"] == "Yes"][merge_key].itertuples(index=False, name=None)
    )
    if "ai_decision_base" in merged.columns:
        still_yes = len(baseline_yes_keys & new_yes_keys)
        reg_pct = (100.0 * still_yes / len(baseline_yes_keys)) if baseline_yes_keys else 0.0
        print(f"\nRegression: baseline Yes count = {len(baseline_yes_keys)}")
        print(f"  Still Yes in new export: {still_yes} ({reg_pct:.1f}%)")
        dropped = baseline_yes_keys - new_yes_keys
        if dropped:
            print(f"  WARNING: {len(dropped)} baseline Yes pairs are NOT Yes in new (investigate)")
            rev = (
                pd.DataFrame(list(dropped), columns=list(merge_key))
                .merge(n, on=merge_key, how="left")
            )
            print("\n--- Regressions: new ai_decision ---")
            if "ai_decision" in rev.columns:
                print(rev["ai_decision"].value_counts().to_string())
            if "triage_pass" in rev.columns:
                print("\n--- Regressions: triage_pass ---")
                print(rev["triage_pass"].value_counts(dropna=False).to_string())
    else:
        print("\n(Merge produced no ai_decision_base — check baseline file IDs.)")

    if "ai_decision_base" in merged.columns:
        resc = merged[(merged["ai_decision_new"] == "Yes") & (merged["ai_decision_base"] == "Filtered")]
        print(f"\nNew Yes that were baseline Filtered: {len(resc)} rows (showing top {args.top})")
        cols = [c for c in ["company_name", "opportunity_name", "final_score", "triage_reason"] if c in resc.columns]
        if cols:
            print(resc.sort_values("final_score", ascending=False)[cols].head(args.top).to_string(index=False))

    out_path = args.write_regression_xlsx.strip()
    if out_path:
        nrow = write_regression_workbook(b, n, merge_key, out_path)
        print(f"\nRegression workbook ({nrow} pair rows): {out_path}")
        print("  sheets: regressed_pairs, by_triage_reason")


if __name__ == "__main__":
    main()
