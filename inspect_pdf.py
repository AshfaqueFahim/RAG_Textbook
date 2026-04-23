import fitz
from pathlib import Path

PDF_PATH = Path(__file__).resolve().parent / "data" / "raw" / "textbook.pdf"
PAGE_RANGE = range(160, 161)  # page indices (0-based); page_num = index + 1

doc = fitz.open(PDF_PATH)

for page_index in PAGE_RANGE:
    page = doc[page_index]
    page_num = page_index + 1
    page_height = page.rect.height
    page_width = page.rect.width

    print(f"\n{'=' * 60}")
    print(f"PAGE {page_num}  (height={page_height:.1f}, width={page_width:.1f})")
    print(f"{'=' * 60}")

    blocks = page.get_text("dict")["blocks"]
    text_blocks = [b for b in blocks if b.get("type") == 0]
    text_blocks.sort(key=lambda b: b["bbox"][1])

    for block in text_blocks:
        bbox = block["bbox"]
        mid_x = (bbox[0] + bbox[2]) / 2
        y0 = bbox[1]
        print(f"  BLOCK bbox={bbox}  mid_x={mid_x:.1f}  y0={y0:.1f}")
        for line in block.get("lines", []):
            line_text = " ".join(s.get("text", "") for s in line.get("spans", []))
            print(f"    line: {repr(line_text.strip())}")
            for span in line.get("spans", []):
                print(f"      span: color={span.get('color')}  size={round(span.get('size', 0), 2)}  "
                      f"text={repr(span.get('text', ''))}")

doc.close()
