import fitz
from pathlib import Path
PDF_PATH = Path(__file__).resolve().parent / "data" / "raw" / "textbook.pdf"
doc = fitz.open(PDF_PATH)
page = doc[174]  # 
print(page.get_text("text"))
doc.close()