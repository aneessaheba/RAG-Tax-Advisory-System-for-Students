"""Step 1: Extract text from PDFs listed in document_manifest.csv into JSON files."""
import os
import csv
import json
import pymupdf  # PyMuPDF - use this instead of 'import fitz' to avoid conflicts

BASE_DIR = os.path.dirname(__file__)
MANIFEST_PATH = os.path.join(BASE_DIR, 'document_manifest.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data_work', 'parsed_docs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    """Extract text from each page of a PDF."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            pages.append({'page_number': page_num + 1, 'text': text})
    return pages


def main():
    with open(MANIFEST_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf_path = os.path.join(BASE_DIR, row['folder'], row['filename'])
            if not os.path.isfile(pdf_path):
                print(f"SKIP - not found: {pdf_path}")
                continue

            print(f"Extracting: {row['doc_id']} ({row['filename']})")
            pages = extract_text_from_pdf(pdf_path)

            output = {
                'doc_id': row['doc_id'],
                'source_type': row['source_type'],
                'title': row['title'],
                'year': row['year'],
                'country': row['country'],
                'pages': pages,
            }

            out_path = os.path.join(OUTPUT_DIR, f"{row['doc_id']}.json")
            with open(out_path, 'w', encoding='utf-8') as out_f:
                json.dump(output, out_f, ensure_ascii=False, indent=2)
            print(f"  -> Saved: {out_path}")


if __name__ == "__main__":
    main()
