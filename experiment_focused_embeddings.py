#!/usr/bin/env python3
"""Experiment + safety check for capability-focused embeddings.

FULL     = what the pipeline embeds today (entire profile; every opp field).
FOCUSED  = same text with only junk PHRASES removed (founding clause, HQ,
           year, ticker; opportunity prices/tonnages/geography). Phrase-level,
           so it never deletes a whole sentence and never drops a capability
           noun.

Safety checks printed:
  1. Sample FULL vs FOCUSED text for companies across sectors (eyeball that
     capabilities survive).
  2. Per-company Spearman rank correlation of opportunity ordering, FULL vs
     FOCUSED, per sector — high = focusing sharpens, does not scramble.
  3. Separation change per company sector.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")
from openai import OpenAI  # noqa: E402

MODEL = "text-embedding-3-large"

# Junk PHRASES only (never whole sentences). Each removes corporate trivia while
# leaving any capability nouns in the same sentence intact.
JUNK = [
    r"\bwas founded by [^.,;]+", r"\bfounded by [^.,;]+",
    r"\b(founded|established|incorporated)\s+(in\s+)?(18|19|20)\d{2}\b",
    r"\b(in|since)\s+(18|19|20)\d{2}\b", r"\b(18|19|20)\d{2}\b",
    r"\bis\s+headquartered\s+in\s+[^.,;]+", r"\bheadquartered\s+in\s+[^.,;]+",
    r"\bis\s+(a\s+)?(global,?\s+)?publicly[- ]traded\b", r"\bpublicly[- ]traded\b",
    r"\blisted on\s+[^.,;]+", r"\b(nasdaq|nyse)\b", r"\bstock exchange\b",
    r"\bengages in the provision of\b", r"\bwas incorporated\b",
]
JUNK_RE = re.compile("|".join(JUNK), re.I)
PRICE = re.compile(r"\b\d[\d,\.]*\s*(usd|ton|tons|%|percent|billion|million|sar|kg)?\b", re.I)
GEO = re.compile(
    r"\b(saudi|arabia|riyadh|jeddah|dammam|kingdom|mena|gcc|gulf|middle east|"
    r"asia[- ]?pacific|europe|africa|americas?|neom|sabic|tasnee|maaden|aramco|olayan)\b", re.I)


def company_focus(name, profile, products, sector) -> str:
    txt = f"{sector}. {profile}. {products}"
    txt = JUNK_RE.sub(" ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def opp_focus(row) -> str:
    fields = [row["What is the opportunity name?"], row["Sector"],
              row["What is the opportunity description?"],
              row["What are the investment highlights?"],
              row["What is the value proposition of this opportunity?"],
              row["What materials are involved or required in the project?"]]
    txt = " ".join(str(f or "") for f in fields)
    txt = GEO.sub(" ", PRICE.sub(" ", txt))
    return re.sub(r"\s+", " ", txt).strip()


def embed(client, texts):
    out = []
    for i in range(0, len(texts), 96):
        r = client.embeddings.create(model=MODEL, input=texts[i:i + 96])
        out.extend(np.asarray(d.embedding, dtype=np.float32) for d in r.data)
    return np.vstack(out)


def cosm(a, b):
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a @ b.T


def main():
    c = pd.read_excel("Data/companies.xlsx")
    o = pd.read_excel("Data/new_opportunities.xlsx")
    client = OpenAI(timeout=60, max_retries=3)

    c_full = (c[["Company Name", "Company Profile", "Product/Services"]]
              .astype(str).agg(" ".join, axis=1)).tolist()
    o_full = o.astype(str).agg(" ".join, axis=1).tolist()
    c_foc = [company_focus(r["Company Name"], r["Company Profile"], r["Product/Services"], r["Sector"])
             for _, r in c.iterrows()]
    o_foc = [opp_focus(r) for _, r in o.iterrows()]

    # 1) Eyeball capability survival on a spread of sectors.
    print("=" * 84)
    print("SAFETY CHECK 1 - did capability text survive the stripping? (sample per sector)")
    print("=" * 84)
    samples = ["Belden", "Chemtrade", "MAADEN", "AL GURG AUTOMATION AND CONTROLS LLC"]
    for nm in samples:
        m = c[c["Company Name"].str.strip() == nm]
        if not len(m):
            continue
        i = m.index[0]
        print(f"\n--- {nm} [{c.loc[i,'Sector']}] ---")
        print("  FULL   :", c_full[i][:150].replace("\n", " "))
        print("  FOCUSED:", c_foc[i][:150].replace("\n", " "))

    print("\nEmbedding full + focused sets...")
    Cf, Of = embed(client, c_full), embed(client, o_full)
    Cx, Ox = embed(client, c_foc), embed(client, o_foc)
    full, foc = cosm(Cf, Of), cosm(Cx, Ox)

    # 2) Does focusing scramble any company's ordering? Spearman per company.
    def spearman(a, b):
        ra = pd.Series(a).rank(); rb = pd.Series(b).rank()
        return ra.corr(rb)
    c["rho"] = [spearman(full[i], foc[i]) for i in range(len(c))]
    print("\n" + "=" * 84)
    print("SAFETY CHECK 2 - ordering correlation FULL vs FOCUSED, by company sector")
    print("  (near 1.0 = focusing SHARPENS the same matches, does not reorder)")
    print("=" * 84)
    bysec = c.groupby("Sector")["rho"].agg(["mean", "min", "count"]).sort_values("mean")
    print(bysec.to_string())
    print(f"\n  worst single company: rho={c['rho'].min():.3f} "
          f"({c.loc[c['rho'].idxmin(),'Company Name']})")

    # 3) Separation change per company sector (mean cosine to best 3 opps minus
    #    mean to the rest — a proxy for how cleanly each company points at its
    #    real targets). Higher is better; we want no sector to get worse.
    def sep(m, i):
        s = np.sort(m[i])[::-1]
        return s[:3].mean() - s[3:].mean()
    c["sep_full"] = [sep(full, i) for i in range(len(c))]
    c["sep_foc"] = [sep(foc, i) for i in range(len(c))]
    print("\n" + "=" * 84)
    print("SAFETY CHECK 3 - top-match separation by company sector: FULL -> FOCUSED")
    print("=" * 84)
    g = c.groupby("Sector")[["sep_full", "sep_foc"]].mean()
    g["change"] = g["sep_foc"] - g["sep_full"]
    for sec, r in g.iterrows():
        flag = "OK" if r["change"] >= -0.002 else "WORSE"
        print(f"  {sec:34} {r['sep_full']:.3f} -> {r['sep_foc']:.3f}  ({r['change']:+.3f})  {flag}")
    print(f"\n  overall mean cosine: FULL {full.mean():.3f} -> FOCUSED {foc.mean():.3f}")


if __name__ == "__main__":
    main()
