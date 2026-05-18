"""
Process all PDF files from Investment Playbook directory.
"""

from pathlib import Path
import pandas as pd

try:
    import PyPDF2
except ImportError:
    print("Installing PyPDF2...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'PyPDF2'])
    import PyPDF2

PDF_DIR = Path("/Users/joudathashmi/Documents/RB Feras and Ismail/Investment Playbook")
OUTPUT_PATH = Path(__file__).parent / "Data" / "investment_playbook_data.xlsx"

def extract_pdf_text(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path.name}: {e}")
    return text

def process_all_pdfs():
    """Process all PDFs in the directory."""
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files\n")
    
    data = []
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        text = extract_pdf_text(pdf_path)
        
        category = pdf_path.stem.split("_")[0] if "_" in pdf_path.stem else "Telecom/ICT"
        
        data.append({
            "Filename": pdf_path.name,
            "Category": category,
            "Opportunity": pdf_path.stem,
            "Text_Content": text,
            "Text_Length": len(text),
            "Page_Count": text.count('\n')
        })
    
    df = pd.DataFrame(data)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False, engine='openpyxl')
    
    print(f"\n{'='*60}")
    print(f"✓ Processed {len(pdf_files)} PDFs successfully")
    print(f"✓ Output saved to: {OUTPUT_PATH}")
    print(f"\nSummary by category:")
    print(df.groupby("Category")["Filename"].count().to_string())
    print(f"\nTotal text extracted: {df['Text_Length'].sum():,} characters")

if __name__ == "__main__":
    process_all_pdfs()
