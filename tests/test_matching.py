#!/usr/bin/env python3
"""Unit tests for matching_v2 pure logic.

No API and no data files needed. Run either way:
    python3 tests/test_matching.py      # plain runner, prints PASS/FAIL
    python3 -m pytest tests/            # if pytest is installed
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matching_v2 as m  # noqa: E402


# ------------------------------- percentile_rank -------------------------------

def test_percentile_rank_monotonic_and_bounded():
    x = np.array([[0.1, 0.9], [0.5, 0.3]])
    p = m.percentile_rank(x)
    assert p.shape == x.shape
    assert p.min() >= 0.0 and p.max() <= 1.0
    # the largest input must get the largest percentile
    assert p.flatten()[x.flatten().argmax()] == p.max()
    assert p.flatten()[x.flatten().argmin()] == p.min()


# --------------------------------- focus text ----------------------------------

def test_focus_company_strips_boilerplate_keeps_capability():
    txt = ("Acme Corp was founded in 1990 and is headquartered in Berlin. "
           "The company produces sulfuric acid and water-treatment chemicals.")
    out = m.focus_company_text(txt)
    assert "1990" not in out
    assert "headquartered" not in out
    # capability nouns survive
    assert "sulfuric acid" in out
    assert "water-treatment chemicals" in out


def test_focus_opportunity_strips_prices_and_geography():
    txt = "Assembly hub in Saudi Arabia. Copper 9,800 USD/ton, supplied by SABIC."
    out = m.focus_opportunity_text(txt)
    assert "saudi" not in out.lower()
    assert "sabic" not in out.lower()
    assert "9,800" not in out and "9800" not in out
    assert "assembly hub" in out.lower()


# -------------------------------- sector_score ---------------------------------

def test_sector_score_exact_match():
    score, label, _, bridge = m.sector_score("ICT Hardware", "ICT Hardware", set())
    assert score == 1.0 and label == "Exact" and bridge is None


def test_sector_score_bridge_fires_with_two_specific_terms():
    # Industrial -> ICT bridge needs >=2 capability terms, one non-generic.
    score, label, _, bridge = m.sector_score(
        "Industrial Manufacturing", "ICT Hardware", {"electronics", "cables"})
    assert bridge == "Industrial ↔ ICT"
    assert score >= 0.5 and label in ("Moderate", "Strong")


def test_sector_score_bridge_does_not_fire_on_generic_only():
    # "assembly" alone is generic and a single term -> no bridge.
    score, label, _, bridge = m.sector_score(
        "Industrial Manufacturing", "ICT Hardware", {"assembly"})
    assert bridge is None


def test_sector_score_no_overlap():
    score, label, _, bridge = m.sector_score("Mining", "Pharmaceutical", set())
    assert bridge is None and label in ("No", "Weak")


# -------------------------------- decision_label -------------------------------

def test_decision_label_gpt_governs():
    assert m.decision_label(0.9, True, True, gpt_decision="Direct") == "High Fit"
    assert m.decision_label(0.4, True, True, gpt_decision="Direct") == "Good Fit"
    assert m.decision_label(0.9, True, True, gpt_decision="Partial") == "Partner Fit"
    assert m.decision_label(0.9, True, True, gpt_decision="No") == "Low Fit"


def test_decision_label_unqualified_is_low():
    assert m.decision_label(0.99, False, True) == "Low Fit"


def test_decision_label_bridge_only_caps_below_high():
    # no GPT verdict, only a bridge link: never High Fit on score alone
    assert m.decision_label(0.95, True, True, bridge_only=True) == "Good Fit"
    assert m.decision_label(0.60, True, True, bridge_only=True) == "Review Needed"


def test_decision_label_score_ladder_without_gpt():
    assert m.decision_label(0.90, True, True) == "High Fit"
    assert m.decision_label(0.75, True, True) == "Good Fit"
    assert m.decision_label(0.55, True, True) == "Review Needed"
    assert m.decision_label(0.40, True, True) == "Low Fit"


# ------------------------------- gate aggregation ------------------------------

class _Completions:
    def __init__(self, outer):
        self.outer = outer

    def create(self, **kw):
        c = self.outer._contents[self.outer._i % len(self.outer._contents)]
        self.outer._i += 1
        message = type("M", (), {"content": c})()
        choice = type("Ch", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()


class FakeClient:
    def __init__(self, contents):
        self._contents = contents
        self._i = 0
        self.chat = type("Chat", (), {"completions": _Completions(self)})()


def _pair():
    comp = pd.Series({"company_name": "C", "Sector": "ICT",
                      "company_profile": "p", "product and Services": "x"})
    opp = pd.Series({"What is the opportunity name?": "O", "Sector": "ICT Hardware",
                     "What is the opportunity description?": "d",
                     "What are the investment highlights?": "h",
                     "What are the key demand drivers?": "dd",
                     "What materials are involved or required in the project?": "mm"})
    return comp, opp


def _json(fit, conf=0.8):
    return '{"fit": "%s", "confidence": %s, "explanation": "e"}' % (fit, conf)


def test_gate_majority_direct():
    comp, opp = _pair()
    client = FakeClient([_json("Direct"), _json("Direct"), _json("None")])
    fit, conf, expl, model, agree = m.gpt_validate(client, ["gpt"], comp, opp,
                                                   votes=3, escalate=False)
    assert fit == "Direct" and agree == "2/3"


def test_gate_three_way_tie_resolves_conservative():
    comp, opp = _pair()
    client = FakeClient([_json("Direct"), _json("Partial"), _json("None")])
    fit, conf, expl, model, agree = m.gpt_validate(client, ["gpt"], comp, opp,
                                                   votes=3, escalate=False)
    assert fit == "No"  # a 1/1/1 split resolves DOWN


def test_gate_majority_partial():
    comp, opp = _pair()
    client = FakeClient([_json("Partial"), _json("Partial"), _json("Direct")])
    fit, conf, expl, model, agree = m.gpt_validate(client, ["gpt"], comp, opp,
                                                   votes=3, escalate=False)
    assert fit == "Partial" and agree == "2/3"


def test_gate_unanimous_first_round_does_not_escalate():
    comp, opp = _pair()
    client = FakeClient([_json("Partial")] * 3)
    fit, conf, expl, model, agree = m.gpt_validate(client, ["gpt"], comp, opp,
                                                   votes=3, escalate=True)
    assert fit == "Partial" and agree == "3/3"
    assert client._i == 3  # no extra calls drawn


def test_gate_split_first_round_escalates_to_five():
    comp, opp = _pair()
    # first 3 split 2/1, escalation draws 2 more (cycled from the start)
    client = FakeClient([_json("Direct"), _json("Direct"), _json("None"),
                         _json("Direct"), _json("Direct")])
    fit, conf, expl, model, agree = m.gpt_validate(client, ["gpt"], comp, opp,
                                                   votes=3, escalate=True)
    assert fit == "Direct" and agree == "4/5"
    assert client._i == 5


# ------------------------------ entity resolution ------------------------------

def test_canonical_name_merges_known_duplicates():
    assert m.canonical_name("Tuwaiq Casting & forging") == m.canonical_name("Tuwaiq Casting and Forging")
    assert m.canonical_name("AL GURG AUTOMATION AND CONTROLS LLC") == m.canonical_name("Al Gurg Automation and Controls")
    # different entities stay distinct
    assert m.canonical_name("Maschinenfabrik Reinhausen GmbH") != m.canonical_name("Reinhausen Middle East")


def test_canonical_name_strips_legal_suffixes_only_at_end():
    assert m.canonical_name("Acme Co Ltd") == "acme"
    assert m.canonical_name("Co Op Industrial") == "co op industrial"


# ---------------------------- consortium readiness -----------------------------

def test_quote_in_text_verifies_normalized_quotes():
    text = "The facility requires precision assembly of hot-swappable PSUs, and RF cabling."
    assert m.quote_in_text("precision assembly of hot-swappable PSUs", text)
    # punctuation/case differences still verify
    assert m.quote_in_text("Precision assembly, of hot swappable psus", text)


def test_quote_in_text_rejects_invented_or_short_quotes():
    text = "The facility requires precision assembly of PSUs."
    assert not m.quote_in_text("GMP-certified biologics fill-finish lines", text)  # invented
    assert not m.quote_in_text("precision assembly", text)  # too short to ground a need


# ----------------------------- human-in-the-loop --------------------------------

def test_load_human_reviews_parses_and_maps_verdicts(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "human_reviews.csv")
        pd.DataFrame([
            {"company": "Belden", "opportunity": "5G Macro", "verdict": "agree"},
            {"company": "Acme", "opportunity": "MRI", "verdict": "Not a fit"},
            {"company": "Junk", "opportunity": "X", "verdict": "maybe"},  # ignored
        ]).to_csv(p, index=False)
        out = m.load_human_reviews(p)
    assert out[("Belden", "5G Macro")] == 1
    assert out[("Acme", "MRI")] == 0
    assert ("Junk", "X") not in out


def test_load_human_reviews_missing_file_is_empty():
    assert m.load_human_reviews("/nonexistent/nowhere.csv") == {}


def test_apply_human_overrides_disagree_and_agree():
    df = pd.DataFrame([
        {"company": "A", "opportunity": "O1", "validated_fit": True,
         "ai_decision": "Partner Fit"},   # analyst rejects
        {"company": "B", "opportunity": "O1", "validated_fit": False,
         "ai_decision": "Low Fit"},       # analyst validates
        {"company": "C", "opportunity": "O1", "validated_fit": True,
         "ai_decision": "Partner Fit"},   # untouched
    ])
    changed = m.apply_human_overrides(df, {("A", "O1"): 0, ("B", "O1"): 1})
    assert changed == 2
    assert not df.loc[0, "validated_fit"] and df.loc[0, "ai_decision"] == "Low Fit"
    assert df.loc[1, "validated_fit"] and df.loc[1, "ai_decision"] == "Partner Fit"
    assert df.loc[2, "validated_fit"] and df.loc[2, "ai_decision"] == "Partner Fit"


# ------------------------------- anti-drift --------------------------------

def test_rubric_hash_is_short_and_stable():
    assert isinstance(m.RUBRIC_HASH, str) and len(m.RUBRIC_HASH) == 8
    import hashlib
    assert m.RUBRIC_HASH == hashlib.md5(
        (m.GPT_SYSTEM_PROMPT + m.RUBRIC_CORE + m.EXEMPLAR_HEADER).encode()
    ).hexdigest()[:8]


def test_build_exemplar_lines_formats_and_keys():
    companies = pd.DataFrame([
        {"company_name": "Belden", "product and Services": "cables and connectivity"},
    ])
    lines = m.build_exemplar_lines(
        {("Belden", "5G Macro"): 1, ("Ghost Co", "X"): 0}, companies)
    d = dict(lines)
    assert "FIT" in d[("Belden", "5G Macro")] and "cables" in d[("Belden", "5G Macro")]
    assert "NOT A FIT" in d[("Ghost Co", "X")]  # unknown company still usable


def test_compute_verdict_flips_detects_changes_only():
    df = pd.DataFrame([
        {"company": "A", "opportunity": "O", "gpt_decision": "No",
         "gpt_agreement": "3/3", "human_verdict": ""},
        {"company": "B", "opportunity": "O", "gpt_decision": "Partial",
         "gpt_agreement": "3/3", "human_verdict": ""},
        {"company": "C", "opportunity": "O", "gpt_decision": "",
         "gpt_agreement": "", "human_verdict": ""},  # ungraded: ignored
    ])
    prior = {("A", "O"): {"decision": "Partial", "ts": "t1", "rubric": "old00000"},
             ("B", "O"): {"decision": "Partial", "ts": "t1", "rubric": m.RUBRIC_HASH}}
    flips = m.compute_verdict_flips(df, prior)
    assert len(flips) == 1
    f = flips[0]
    assert f["company"] == "A" and f["previous_verdict"] == "Partial"
    assert f["new_verdict"] == "No" and f["rubric_changed"] == "yes"


# ------------------------------ v3 investment engine ---------------------------

import matching_v3 as v3  # noqa: E402


def test_v3_sector_similarity_no_more_zeros():
    # the spec's broken case: different naming conventions must not yield 0
    assert v3.sector_similarity("industrial manufacturing", "ict hardware") == 0.45
    assert v3.sector_similarity("industrial chemicals", "pharmaceutical manufacturing") == 0.55
    assert v3.sector_similarity("biotechnology", "pharmaceutical manufacturing") == 0.65
    assert v3.sector_similarity("ict hardware", "ict hardware") == 1.0
    # parent-child
    assert v3.sector_similarity("industrial", "industrial manufacturing") == 0.80
    # genuinely unrelated stays low
    assert v3.sector_similarity("mining & minerals", "pharmaceutical manufacturing") <= 0.2


def test_v3_sector_synonyms():
    assert v3.normalize_sector_label("MedTech") == "medical devices"
    assert v3.normalize_sector_label("Pharmaceutical") == "pharmaceutical manufacturing"
    assert v3.normalize_sector_label("Oil, Gas, Energy & Water") == "energy"


def test_v3_value_chain_spec_examples():
    # Drug Developer vs: chemical supplier Low, biotech Very High, API mfr Medium
    assert v3.value_chain_score(["Developer"], "Raw Material Supplier") <= 0.2
    assert v3.value_chain_score(["Developer"], "Research Company") >= 0.85
    assert 0.4 <= v3.value_chain_score(["Developer"], "Contract Manufacturer") <= 0.6


def test_v3_penalties_stack_on_supplier_to_developer():
    factor, names = v3.compute_penalties(0.2, 0.4, 0.8, 0.15,
                                         "Raw Material Supplier", ["Developer"])
    assert "supplier_to_developer" in names and "sector_mismatch" in names
    assert factor < 0.5  # multiple penalties compound


def test_v3_decide_hierarchy():
    assert v3.decide(0.8, "No", "", True) == "Weak Match"       # gate rejects
    assert v3.decide(0.8, "Partial", "", True) == "Strong Match"  # supplier caps below Excellent
    assert v3.decide(0.2, "No", "Agree", True) == "Good Match"  # analyst floor
    assert v3.decide(0.9, "", "Disagree", False) == "Poor Match"  # analyst kill
    assert v3.decide(0.9, "", "", False) == "Potential Match"   # unvetted ceiling


def test_v3_confidence_bounds():
    c = v3.confidence_score(600, 1500, 0.8, [0.5, 0.55, 0.5, 0.6, 0.5], "3/3")
    assert 0 <= c <= 100 and v3.confidence_label(c) in ("High", "Medium", "Low")


# ---------------------------------- runner -------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}  {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}  {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
