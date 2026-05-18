#!/usr/bin/env python3
"""
Fill missing investor emails in an Excel file.

Usage:
  python add_emails_to_investors_profiles.py \
    --input "/path/to/Investors_Profiles_7th_March_V.xlsx"

Optional:
  --output "/path/to/output.xlsx"
  --sheet "Sheet1"
  --no-openai         # only regex extraction, no model calls
  --limit 50          # process first N rows only

This script uses the same key style as your existing code:
  OPENAI_API_KEY from environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI

EMAIL_REGEX = re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
DEFAULT_MODEL = "gpt-4o-mini"


def normalize_email(value: str) -> str:
    email = value.strip().strip(".,;:()[]<>\"'").lower()
    if re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email):
        return email
    return ""


def first_email_in_text(text: str) -> str:
    if not text:
        return ""
    match = EMAIL_REGEX.search(text)
    return normalize_email(match.group(1)) if match else ""


def normalize_key(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_email_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"email", "e-mail", "mail", "email address", "work email"}:
            return col
    return "Email"


def row_text_for_lookup(row: pd.Series) -> str:
    pieces = []
    preferred = [
        "Investor Name",
        "Name",
        "Company",
        "Organization",
        "Website",
        "LinkedIn",
        "Profile",
        "Description",
        "Bio",
        "File Name",
        "Filename",
    ]
    existing = {str(c).strip().lower(): c for c in row.index}
    for label in preferred:
        col = existing.get(label.lower())
        if col is not None:
            value = row.get(col, "")
            if pd.notna(value) and str(value).strip():
                pieces.append(f"{label}: {value}")

    # Fallback: include all columns if preferred fields are empty
    if not pieces:
        for col, value in row.items():
            if pd.notna(value) and str(value).strip():
                pieces.append(f"{col}: {value}")

    return "\n".join(pieces)


def ask_openai_for_email(client: OpenAI, context_text: str, model: str) -> str:
    prompt = f"""
You are given one investor/company record.
Return only valid business email if confidently inferable from the context.
If no reliable email is available, return empty string.

Output must be strict JSON:
{{"email": "<email_or_empty>"}}

Context:
{context_text}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Extract a reliable email only. No guessing without support.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    content = (response.choices[0].message.content or "").strip()
    if not content:
        return ""

    try:
        data = json.loads(content)
        return normalize_email(str(data.get("email", "")))
    except Exception:
        # Fallback in case model returns plain text with an email
        return first_email_in_text(content)


def _apply_source_folder_matches(df: pd.DataFrame, email_col: str, source_folder: str) -> int:
    """
    Fill missing target emails by exact Full Name / Slug Name matches against rows
    in spreadsheet files under source_folder that contain emails.
    """
    root = Path(source_folder)
    if not root.exists():
        # Graceful fallback for folder names that differ only by trailing spaces.
        source_key = source_folder.strip().lower()
        siblings = [p for p in Path.cwd().iterdir() if p.is_dir() and p.name.strip().lower() == source_key]
        if len(siblings) == 1:
            root = siblings[0]
    if not root.exists() or not root.is_dir():
        print(f"Source folder not found or not a directory: {source_folder}")
        return 0

    if "Full Name" not in df.columns and "Slug Name" not in df.columns:
        print("Skipping source-folder matching: 'Full Name'/'Slug Name' not present in target file.")
        return 0

    missing_rows = df.index[df[email_col].fillna("").astype(str).str.strip().eq("")]
    key_to_indices: dict[str, list[int]] = {}
    for idx in missing_rows:
        full_name = normalize_key(df.at[idx, "Full Name"]) if "Full Name" in df.columns else ""
        slug_name = normalize_key(df.at[idx, "Slug Name"]) if "Slug Name" in df.columns else ""
        if len(full_name) >= 4:
            key_to_indices.setdefault(full_name, []).append(idx)
        if len(slug_name) >= 4:
            key_to_indices.setdefault(slug_name, []).append(idx)

    if not key_to_indices:
        return 0

    source_file_count = 0
    scanned_rows = 0
    source_match_count = 0
    name_col_tokens = ("name", "contact", "investor", "person", "guest", "profile")

    for file_path in root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in {".xlsx", ".csv"}:
            continue

        source_file_count += 1
        try:
            if file_path.suffix.lower() == ".csv":
                sheets = [pd.read_csv(file_path, dtype=str, encoding_errors="ignore")]
            else:
                xls = pd.ExcelFile(file_path)
                sheets = []
                for sheet in xls.sheet_names:
                    try:
                        sheets.append(pd.read_excel(file_path, sheet_name=sheet, dtype=str))
                    except Exception:
                        continue

            for source_df in sheets:
                if source_df is None or source_df.empty:
                    continue

                columns = [str(c) for c in source_df.columns]
                email_cols = [c for c in columns if "email" in c.lower()]
                person_cols = [c for c in columns if any(tok in c.lower() for tok in name_col_tokens)]

                first_cols = [c for c in columns if c.lower().strip() in {"first name", "firstname", "first"}]
                last_cols = [c for c in columns if c.lower().strip() in {"last name", "lastname", "last"}]

                for _, source_row in source_df.iterrows():
                    scanned_rows += 1

                    emails: list[str] = []
                    for col in email_cols:
                        emails.extend(EMAIL_REGEX.findall(str(source_row.get(col, "") or "")))
                    if not emails:
                        emails.extend(EMAIL_REGEX.findall(" | ".join(str(v) for v in source_row.tolist() if pd.notna(v))))
                    if not emails:
                        continue
                    email = normalize_email(emails[0])
                    if not email:
                        continue

                    candidate_keys: list[str] = []
                    for col in person_cols:
                        key = normalize_key(source_row.get(col, ""))
                        if len(key) >= 4:
                            candidate_keys.append(key)

                    if first_cols and last_cols:
                        for first_col in first_cols:
                            for last_col in last_cols:
                                full_key = normalize_key(f"{source_row.get(first_col, '')} {source_row.get(last_col, '')}")
                                if len(full_key) >= 4:
                                    candidate_keys.append(full_key)

                    for key in dict.fromkeys(candidate_keys):
                        idx_candidates = key_to_indices.get(key, [])
                        if not idx_candidates:
                            continue
                        selected_idx = None
                        for idx in idx_candidates:
                            if not normalize_email(str(df.at[idx, email_col])):
                                selected_idx = idx
                                break
                        if selected_idx is None:
                            continue
                        df.at[selected_idx, email_col] = email
                        source_match_count += 1
                        break
        except Exception:
            continue

    print(f"Source-folder scan: {source_file_count} files, {scanned_rows} rows scanned")
    return source_match_count


def fill_emails(
    input_path: str,
    output_path: str,
    sheet_name: Optional[str],
    use_openai: bool,
    limit: Optional[int],
    model: str,
    source_folder: Optional[str],
) -> None:
    sheet_arg = sheet_name if sheet_name is not None else 0
    df = pd.read_excel(input_path, sheet_name=sheet_arg)
    if isinstance(df, dict):
        if not df:
            raise ValueError(f"No sheets found in workbook: {input_path}")
        first_sheet_name = next(iter(df.keys()))
        df = df[first_sheet_name].copy()
        print(f"Loaded first sheet: {first_sheet_name}")

    if limit is not None and limit > 0:
        df = df.head(limit).copy()

    email_col = detect_email_column(df)
    if email_col not in df.columns:
        df[email_col] = ""

    # Ensure string type for updates
    df[email_col] = df[email_col].fillna("").astype(str)

    client = None
    if use_openai:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Export it, then run again.")
        client = OpenAI(api_key=api_key)

    updated_count = 0
    extracted_count = 0
    source_count = 0
    ai_count = 0

    for idx, row in df.iterrows():
        existing = normalize_email(str(row.get(email_col, "")))
        if existing:
            continue

        # 1) Try direct regex extraction from row text
        text_blob = row_text_for_lookup(row)
        email = first_email_in_text(text_blob)
        if email:
            df.at[idx, email_col] = email
            updated_count += 1
            extracted_count += 1
            continue

    # 2) Optional source-folder matching (exact name/slug match only)
    if source_folder:
        source_count = _apply_source_folder_matches(df, email_col, source_folder)
        updated_count += source_count

    # 3) Optional AI step for remaining blanks
    if client is not None:
        for idx, row in df.iterrows():
            existing = normalize_email(str(row.get(email_col, "")))
            if existing:
                continue
            text_blob = row_text_for_lookup(row)
            try:
                ai_email = ask_openai_for_email(client, text_blob, model=model)
            except Exception:
                ai_email = ""
            if ai_email:
                df.at[idx, email_col] = ai_email
                updated_count += 1
                ai_count += 1

    df.to_excel(output_path, index=False)
    print(f"Done. Output saved to: {output_path}")
    print(f"Email column used: {email_col}")
    print(f"Rows updated: {updated_count}")
    print(f" - Extracted via regex: {extracted_count}")
    print(f" - Matched from source folder: {source_count}")
    print(f" - Filled via OpenAI: {ai_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill missing emails in investor profile Excel file.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input Excel file, e.g., Investors_Profiles_7th_March_V.xlsx",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output path. Default: input path with '_with_emails.xlsx' suffix.",
    )
    parser.add_argument("--sheet", default=None, help="Sheet name (optional).")
    parser.add_argument("--no-openai", action="store_true", help="Disable OpenAI fallback.")
    parser.add_argument(
        "--source-folder",
        default="",
        help="Optional folder containing CSV/XLSX files to scan for emails and match by Full Name/Slug Name.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N rows.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path.replace(".xlsx", "_with_emails.xlsx")
    fill_emails(
        input_path=input_path,
        output_path=output_path,
        sheet_name=args.sheet,
        use_openai=not args.no_openai,
        limit=args.limit,
        model=args.model,
        source_folder=(args.source_folder or None),
    )


if __name__ == "__main__":
    main()
