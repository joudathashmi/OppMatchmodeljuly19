#!/usr/bin/env python3
"""Independent verifier for Output/matching_output_v3.csv.

Recomputes every layer of the matching logic FROM SOURCE (input spreadsheets,
embedding cache, enrichment cache, gate labels, human reviews) and diffs the
results against what the output file claims. A format check proves the file is
well-shaped; this proves the numbers are the ones the documented logic
produces.

Layers verified:
  1. Sector similarity   - recomputed from the taxonomy + enrichment cache.
  2. Semantic scores     - profile/product percentile + specificity blend,
                           recomputed from cached embeddings.
  3. Value-chain score   - recomputed from cached roles x compatibility matrix.
  4. Score composition   - final_score must equal the weighted pair evidence
                           times a LEGAL penalty product (penalties form a
                           finite set of possible products).
  5. Decision logic      - decide() re-run per row from the gate verdict
                           (labels jsonl), analyst verdicts and vote depth.
  6. Match type + status - recomputed from the output columns themselves.
  7. Rank + ordering     - per-opportunity priority ranking re-derived.

Run: python3 verify_v3.py            (no GPT calls: all caches must be warm)
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd

import matching_v3 as v3
from matching_v2 import (SPECIFICITY_BLEND, build_vectors, cosine_matrix,
                         load_companies, load_human_reviews,
                         load_opportunities, percentile_rank, resolve_backends)

EPS = 0.0021  # everything is rounded to 3 decimals downstream

PENALTY_FACTORS = [0.65, 0.75, 0.70, 0.70, 0.70]
LEGAL_PRODUCTS = sorted({round(np.prod(c), 6)
                         for r in range(len(PENALTY_FACTORS) + 1)
                         for c in itertools.combinations(PENALTY_FACTORS, r)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=v3.OUTPUT_CSV)
    ap.add_argument("--trace", default="Belden",
                    help="company substring for the end-to-end trace")
    args_ns = argparse.Namespace(no_openai=False, require_openai=False,
                                 embed_blend=0.3, chat_provider="auto")
    try:
        from dotenv import load_dotenv
        load_dotenv(".env")
    except ImportError:
        pass
    args = ap.parse_args()

    out = pd.read_csv(args.csv, keep_default_na=False)
    companies = load_companies()
    opps = load_opportunities()
    human = load_human_reviews()
    backends = resolve_backends(args_ns)

    results = {}

    # ---- layers 1-3 need the enrichment cache and embeddings ----
    comp_enrich, opp_enrich = v3.enrich_all(
        backends["chat_client"], backends["chat_models"], companies, opps)

    prof_mat, prod_mat, opp_mat, mode = build_vectors(companies, opps, args_ns, backends)
    sim_p = cosine_matrix(prof_mat, opp_mat)
    sim_q = cosine_matrix(prod_mat, opp_mat)
    sem_p = ((1 - SPECIFICITY_BLEND) * percentile_rank(sim_p)
             + SPECIFICITY_BLEND * percentile_rank(sim_p - sim_p.mean(axis=1, keepdims=True)))
    sem_q = ((1 - SPECIFICITY_BLEND) * percentile_rank(sim_q)
             + SPECIFICITY_BLEND * percentile_rank(sim_q - sim_q.mean(axis=1, keepdims=True)))

    ci = {n: i for i, n in enumerate(companies["company_name"])}
    oi = {n: j for j, n in enumerate(opps["What is the opportunity name?"])}

    dev_sector = dev_sem_p = dev_sem_q = dev_vc = 0.0
    n_checked = 0
    for r in out.itertuples():
        i, j = ci.get(r.company_name), oi.get(r.opportunity_name)
        if i is None or j is None:
            continue
        n_checked += 1
        ce = comp_enrich.get(r.company_name, {})
        oe = opp_enrich.get(r.opportunity_name, {})
        c_node = ce.get("normalized_sector") or v3.normalize_sector_label(
            companies.iloc[i]["Sector"])
        o_node = oe.get("normalized_sector") or v3.normalize_sector_label(
            opps.iloc[j]["Sector"])
        dev_sector = max(dev_sector, abs(v3.sector_similarity(c_node, o_node)
                                         - float(r.sector_similarity)))
        dev_sem_p = max(dev_sem_p, abs(float(sem_p[i, j]) - float(r.profile_similarity)))
        dev_sem_q = max(dev_sem_q, abs(float(sem_q[i, j]) - float(r.product_similarity)))
        vc = v3.value_chain_score(oe.get("required_roles", []) or [],
                                  ce.get("value_chain_role", ""),
                                  ce.get("secondary_role", ""))
        dev_vc = max(dev_vc, abs(vc - float(r.value_chain_score)))

    results[f"1. sector_similarity recomputed ({n_checked} rows, max dev {dev_sector:.4f})"] = dev_sector <= EPS
    results[f"2a. profile_similarity recomputed (max dev {dev_sem_p:.4f})"] = dev_sem_p <= EPS
    results[f"2b. product_similarity recomputed (max dev {dev_sem_q:.4f})"] = dev_sem_q <= EPS
    results[f"3. value_chain_score recomputed (max dev {dev_vc:.4f})"] = dev_vc <= EPS

    # ---- layer 4: score composition and legal penalty products ----
    w = {k: val for k, val in v3.DEFAULT_WEIGHTS.items() if val > 0}
    s = sum(w.values())
    w = {k: val / s for k, val in w.items()}
    bad_comp = []
    for r in out.itertuples():
        base = (w["sector"] * float(r.sector_similarity)
                + w["profile"] * float(r.profile_similarity)
                + w["product"] * float(r.product_similarity))
        final = float(r.final_score)
        if final > base + EPS and abs(final - 0.05) > EPS:
            bad_comp.append((r.company_name, r.opportunity_name, "final above evidence"))
            continue
        implied = final / base if base > 1e-9 else 1.0
        legal = any(abs(implied - p) <= 0.02 for p in LEGAL_PRODUCTS) or abs(final - 0.05) <= EPS
        if not legal:
            bad_comp.append((r.company_name, r.opportunity_name,
                             f"implied penalty {implied:.3f} not a legal product"))
    results[f"4. final = pair evidence x legal penalty product ({len(bad_comp)} violations)"] = not bad_comp

    # ---- layer 5: decision logic re-derived from gate labels + human ----
    latest = {}
    try:
        with open(v3.LABELS_JSONL) as fh:
            for line in fh:
                d = json.loads(line)
                latest[(d["company"], d["opportunity"])] = d
    except FileNotFoundError:
        pass
    n_dec = n_dec_bad = n_dec_missing = 0
    examples = []
    for r in out.itertuples():
        lab = latest.get((r.company_name, r.opportunity_name))
        if not lab:
            n_dec_missing += 1
            continue
        hv = {1: "Agree", 0: "Disagree"}.get(
            human.get((r.company_name, r.opportunity_name), -1), "")
        agree = str(lab.get("agreement", ""))
        light = (int(agree.split("/")[1]) <= 1 if "/" in agree
                 else int(lab.get("votes", 3)) <= 1)
        expect = v3.decide(float(r.final_score), lab["decision"], hv, True, light=light)
        n_dec += 1
        if expect != r.decision:
            n_dec_bad += 1
            if len(examples) < 4:
                examples.append(f"{r.company_name}->{r.opportunity_name}: "
                                f"file {r.decision} vs derived {expect}")
    results[f"5. decision re-derived from gate+human ({n_dec} rows, {n_dec_bad} mismatches, {n_dec_missing} unlabeled)"] = (n_dec_bad == 0 and n_dec > 0)
    for e in examples:
        print("   decision mismatch:", e)

    # ---- layer 6: match_type and opportunity_status from the file itself ----
    mt_bad = sum(1 for r in out.itertuples()
                 if v3.match_type(r.decision, float(r.value_chain_score),
                                  r.suggested_localization_model,
                                  getattr(r, "value_chain_role", "")) != r.match_type)
    st = {oid: v3.opportunity_status(list(zip(g["decision"], g["match_type"])))
          for oid, g in out.groupby("opportunity_id")}
    st_bad = sum(1 for r in out.itertuples()
                 if st[r.opportunity_id] != r.opportunity_status)
    results[f"6. match_type + opportunity_status recomputed ({mt_bad}+{st_bad} mismatches)"] = mt_bad == 0 and st_bad == 0

    # ---- layer 7: rank = per-opportunity priority (tier, then score) ----
    tidx = {t: k for k, t in enumerate(v3.TIER_ORDER)}
    rank_ok = True
    for _, g in out.groupby("opportunity_id"):
        g = g.sort_values("rank")
        if sorted(g["rank"]) != list(range(1, len(g) + 1)):
            rank_ok = False
        keys = [(tidx[d], -f) for d, f in zip(g["decision"], g["final_score"])]
        if keys != sorted(keys):
            rank_ok = False
    results["7. rank = tier-then-score priority, contiguous per opportunity"] = rank_ok

    # ---- report ----
    print("\n" + "=" * 76)
    for k, ok in results.items():
        print(("PASS  " if ok else "FAIL  ") + k)
    n_fail = sum(1 for ok in results.values() if not ok)
    print("=" * 76)
    print(f"{len(results) - n_fail}/{len(results)} layers verified")

    # ---- end-to-end trace of one pair ----
    tr = out[out["company_name"].str.contains(args.trace)].head(1)
    if len(tr):
        r = tr.iloc[0]
        i, j = ci[r["company_name"]], oi[r["opportunity_name"]]
        ce = comp_enrich.get(r["company_name"], {})
        oe = opp_enrich.get(r["opportunity_name"], {})
        base = (w["sector"] * r["sector_similarity"]
                + w["profile"] * r["profile_similarity"]
                + w["product"] * r["product_similarity"])
        lab = latest.get((r["company_name"], r["opportunity_name"]), {})
        print(f"\nEND-TO-END TRACE: {r['company_name']} -> {r['opportunity_name']}")
        print(f"  sector: company node '{ce.get('normalized_sector')}' vs "
              f"opportunity node '{oe.get('normalized_sector')}' "
              f"-> similarity {r['sector_similarity']}")
        print(f"  embeddings ({mode}): raw cosines {sim_p[i, j]:.3f}/{sim_q[i, j]:.3f} "
              f"-> percentile+specificity {r['profile_similarity']}/{r['product_similarity']}")
        print(f"  value chain: role '{ce.get('value_chain_role')}' vs needs "
              f"{oe.get('required_roles')} -> {r['value_chain_score']} (informational)")
        print(f"  score: {w['sector']:.3f}x{r['sector_similarity']} + "
              f"{w['profile']:.3f}x{r['profile_similarity']} + "
              f"{w['product']:.3f}x{r['product_similarity']} = {base:.3f} "
              f"-> final {r['final_score']} (penalty factor "
              f"{r['final_score'] / base if base else 1:.3f})")
        hv = {1: "Agree", 0: "Disagree"}.get(
            human.get((r["company_name"], r["opportunity_name"]), -1), "none")
        print(f"  gate: {lab.get('decision', '?')} ({lab.get('agreement', '?')}, "
              f"votes {lab.get('votes', '?')}) | analyst: {hv} "
              f"-> decision {r['decision']}")
        print(f"  type/status: {r['match_type']} | {r['opportunity_status']} | "
              f"rank {r['rank']} | confidence {r['confidence_score']}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
