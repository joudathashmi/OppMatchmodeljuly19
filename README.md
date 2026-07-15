# Matching model (SSDAMM)

Python pipeline for **company–opportunity matching**: sector gating, semantic similarity (embeddings + TF–IDF), GPT validation, ranking, and Excel export.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (see `business_grade_matching.py` docstring).

## Main pipeline

Primary script: **`matching_v2.py`**. See [docs/matching_v2.md](docs/matching_v2.md)
for the full architecture, scoring formula, config knobs, and known issues.

Rough flow:

1. Load + normalize companies and opportunities
2. Vocabulary fit (protected sector vocab, corpus-common suppression, IDF)
3. Vectorize (OpenAI embeddings, or hybrid TF-IDF fallback)
4. Percentile-calibrated cosine similarity + specificity correction
5. Per-pair sector / evidence / soft-match scoring and fusion
6. Ranking in both directions from one scoring table
7. Optional GPT validation on qualified top-N (gates label, logs to `gpt_labels.jsonl`)
8. Export `Output/matches_v2.xlsx` (Opportunity_View, Company_View, All_Pairs, Abstentions, Diagnostics)

```bash
python3 matching_v2.py                  # auto: OpenAI if key works, else TF-IDF
python3 matching_v2.py --no-gpt         # skip GPT validation
python3 matching_v2.py --no-openai      # force TF-IDF fallback
python3 matching_v2.py --require-openai  # fail hard instead of TF-IDF fallback
```

### Legacy pipeline

`business_grade_matching.py` is the older script kept for reference. Its flow:
preprocessing, sector ontology expansion, sector filtering, semantic
embedding/similarity, product/service matching, GPT validation, soft-match mode,
ranking and export. Resume / partial runs:

```bash
python3 business_grade_matching.py --resume-export
python3 business_grade_matching.py --resume-from-step8
```

## Other utilities

| File | Role |
|------|------|
| `sector_inference.py` | Sector vocabulary and company enrichment |
| `matcher_signals.py` | Shared matching helpers (keywords, JSON parsing, labels) |
| `calibration_report.py` | Calibration / reporting |
| `extract_opportunities_structured.py` | Structured opportunity extraction |
| `build_opportunities_xlsx.py` | Build opportunities spreadsheet |
| `add_emails_to_investors_profiles.py` | Investor profile email enrichment |
| `process_investment_pdfs.py` | PDF processing helper |

Notebooks: `Code.ipynb`, `Code.executed.ipynb`.

## Data and ignores

Large or local inputs (spreadsheets, pickles, `Data/`, etc.) are excluded via `.gitignore`. The pipeline expects a company workbook such as **`kpmgfile.xlsx`** in the run directory when you execute the full matching flow.

## License

If you add a license, describe it here.
