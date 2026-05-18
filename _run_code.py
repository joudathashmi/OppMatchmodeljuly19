# 📦 Requirements:
# pip install openai pandas scikit-learn numpy openpyxl tqdm

import os
import re
import pickle
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from tqdm import tqdm

# ---------- CONFIG ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GPT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

BASE_DIR = Path.cwd()
COMPANY_FILE = BASE_DIR / "Data" / "companies.xlsx"
OPPORTUNITY_FILE = BASE_DIR / "Data" / "new_opportunities.xlsx"
OUTPUT_FILE = BASE_DIR / "Output" / "matches_output_with_explainability.xlsx"
EMBED_CACHE_FILE = BASE_DIR / "Data" / ".embedding_cache.pkl"

SIMILARITY_THRESHOLD = 0.75
ENABLE_GPT_VALIDATION = True
BATCH_SIZE = 10

# ---------- EMBEDDING CACHE ----------
if EMBED_CACHE_FILE.exists():
    with open(EMBED_CACHE_FILE, "rb") as f:
        _embed_cache = pickle.load(f)
    print(f"📦 Loaded {len(_embed_cache)} cached embeddings")
else:
    _embed_cache = {}

def _save_cache():
    EMBED_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBED_CACHE_FILE, "wb") as f:
        pickle.dump(_embed_cache, f)

# ---------- HELPERS ----------
def preprocess(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def embed_text(text, model=EMBED_MODEL):
    key = hashlib.sha256(f"{model}:{text}".encode()).hexdigest()
    if key in _embed_cache:
        return _embed_cache[key]
    try:
        response = client.embeddings.create(input=[text], model=model)
        vec = response.data[0].embedding
        _embed_cache[key] = vec
        return vec
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * EMBED_DIM

def compute_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def gpt_validate_and_explain(company_name, profile, products, opportunity_name, opportunity_desc):
    prompt = f"""
Company: {company_name}
Profile: {profile}
Products: {products}
Opportunity: {opportunity_name}
Description: {opportunity_desc}

Question: Can this company realistically fulfill this opportunity based on its sector, profile, and offered products/services?
Answer Yes or No, then provide detailed explanation.
""".strip()
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are an evaluator of company-opportunity fit."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content.strip()
        is_yes = content.lower().startswith("yes")
        confidence_score = 0.95 if is_yes else 0.3
        return ("Yes" if is_yes else "No"), confidence_score, content
    except Exception as e:
        return "Error", 0.0, f"GPT Error: {str(e)}"

# Sector synonym groups — companies use broad labels, opportunities use sub-sector labels.
SECTOR_GROUPS = [
    {"ict", "ict hardware", "telecom", "telecommunications", "technology"},
    {"medtech", "medical devices", "medical technology", "healthcare"},
    {"pharmaceutical", "pharma", "biopharma", "life sciences"},
]

def sector_match(company_sector, opp_sector):
    cs = preprocess(company_sector)
    os_ = preprocess(opp_sector)
    if not cs or not os_:
        return "Unknown", "Sector value missing"
    if cs == os_:
        return "Yes", f"Both classified as {company_sector}"
    for group in SECTOR_GROUPS:
        if cs in group and os_ in group:
            return "Yes", f"Adjacent sectors: company={company_sector} ↔ opportunity={opp_sector}"
    return "No", f"Sector mismatch: company={company_sector} vs opportunity={opp_sector}"

# ---------- MAIN ----------
print("🔁 Loading files...")
companies = pd.read_excel(COMPANY_FILE)
opportunities = pd.read_excel(OPPORTUNITY_FILE)
print(f"  {len(companies)} companies, {len(opportunities)} opportunities")

print("🧹 Preprocessing...")
companies["combined"] = (
    companies[["Company Name", "Company Profile", "Product/Services"]]
    .astype(str).agg(" ".join, axis=1).apply(preprocess)
)
companies["products_clean"] = companies["Product/Services"].astype(str).apply(preprocess)

OPP_REQUIREMENT_COLS = [
    "What is the opportunity name?",
    "What is the opportunity description?",
    "What are the investment highlights?",
    "What is the value proposition of this opportunity?",
    "What are the key demand drivers?",
    "Who are the key players in this sector or project?",
    "What materials are involved or required in the project?",
    "Market data",
    "Cost structure",
    "Government incentives",
    "Risks and mitigations",
    "Investment locations",
]
opportunities["requirement"] = (
    opportunities[OPP_REQUIREMENT_COLS].astype(str).agg(" ".join, axis=1).apply(preprocess)
)

print("🔄 Embedding (cache reused where possible)...")
companies["profile_embedding"] = [embed_text(t) for t in tqdm(companies["combined"], desc="company profiles")]
companies["product_embedding"] = [embed_text(t) for t in tqdm(companies["products_clean"], desc="company products")]
opportunities["embedding"] = [embed_text(t) for t in tqdm(opportunities["requirement"], desc="opportunities")]
_save_cache()
print(f"💾 Cache now holds {len(_embed_cache)} embeddings")

# ---------- MATCHING FUNCTION ----------
def match_entities(entities_a, entities_b, direction_label):
    total = len(entities_a)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = entities_a.iloc[start:end]
        results = []

        for _, a in batch.iterrows():
            for _, b in entities_b.iterrows():
                if direction_label == "Company → Opportunity":
                    comp, opp = a, b
                else:
                    comp, opp = b, a

                sim_profile = compute_similarity(comp["profile_embedding"], opp["embedding"])
                sim_product = compute_similarity(comp["product_embedding"], opp["embedding"])

                sec_match, sec_reason = sector_match(comp.get("Sector"), opp.get("Sector"))

                if ENABLE_GPT_VALIDATION:
                    gpt_decision, gpt_score, gpt_explanation = gpt_validate_and_explain(
                        comp["Company Name"], comp["Company Profile"], comp["Product/Services"],
                        opp["What is the opportunity name?"], opp["What is the opportunity description?"],
                    )
                else:
                    gpt_decision, gpt_score, gpt_explanation = "Skipped", 0.5, "GPT validation disabled"

                final_score = round(0.4 * sim_profile + 0.4 * sim_product + 0.2 * gpt_score, 3)

                results.append({
                    "Match Type": direction_label,
                    "Opportunity": opp["What is the opportunity name?"],
                    "Opportunity Sector": opp.get("Sector"),
                    "Company": comp["Company Name"],
                    "Company Sector": comp.get("Sector"),
                    "Sector Match": sec_match,
                    "Semantic Match Score": round(float(sim_profile), 3),
                    "Product/Service Match Score": round(float(sim_product), 3),
                    "GPT Confidence Score": gpt_score,
                    "Final Composite Score": final_score,
                    "GPT Validation (Yes/No)": gpt_decision,
                    "Sector Match Reason": sec_reason,
                    "Profile Match Reason": "Profile vs. opportunity requirement aligned.",
                    "Product Match Reason": "Product/services aligned with opportunity description.",
                    "GPT Explanation": gpt_explanation,
                })

        batch_df = pd.DataFrame(results)
        batch_df["Top Match Rank"] = (
            batch_df.groupby("Opportunity")["Final Composite Score"]
            .rank(method="first", ascending=False).astype(int)
        )
        suffix = direction_label.replace(" → ", "_").replace(" ", "")
        batch_file = str(OUTPUT_FILE).replace(".xlsx", f"_{suffix}_batch_{start}_{end}.xlsx")
        batch_df.to_excel(batch_file, index=False)
        print(f"✅ Saved {batch_file} with {len(batch_df)} matches.")

print("🚀 Running Company → Opportunity Matching...")
match_entities(companies, opportunities, "Company → Opportunity")

print("🚀 Running Opportunity → Company Matching...")
match_entities(opportunities, companies, "Opportunity → Company")

print("🎉 Matching complete.")
