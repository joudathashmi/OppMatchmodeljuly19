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

Primary script: **`business_grade_matching.py`**

Rough flow (see the module docstring for full detail):

1. Preprocessing  
2. Sector ontology expansion  
3. Sector filtering  
4. Semantic embedding and similarity  
5. Product/service matching (combined scores)  
6. GPT-based validation  
7. Soft match mode (high similarity, relaxed sectors)  
8. Ranking and export  

Typical outputs (under `Output/` when run end-to-end): ranked matches workbook, intermediate pickles for resume.

Resume / partial runs:

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
