import os
import csv
import json
import fitz  # PyMuPDF

RAW_DIR = os.path.join(os.path.dirname(__file__), 'data_raw')
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), 'document_manifest.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data_work', 'parsed_docs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_pdf(pdf_path):
    pages = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                pages.append({
                    'page_number': page_num + 1,
                    'text': text
                })
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return pages

def main():
    with open(MANIFEST_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            doc_id = row['doc_id']
            folder = row['folder']
            filename = row['filename']
            pdf_path = os.path.join(RAW_DIR, folder, filename)
            if not os.path.isfile(pdf_path):
                print(f"PDF not found: {pdf_path}")
                continue
            print(f"Parsing {doc_id}: {pdf_path}")
            pages = parse_pdf(pdf_path)
            output = {
                'doc_id': doc_id,
                'source_type': row['source_type'],
                'title': row['title'],
                'year': row['year'],
                'country': row['country'],
                'file': {
                    'folder': folder,
                    'filename': filename
                },
                'pages': pages
            }
            out_path = os.path.join(OUTPUT_DIR, f"{doc_id}.json")
            try:
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
                print(f"Saved: {out_path}")
            except Exception as e:
                print(f"Error saving {out_path}: {e}")

if __name__ == "__main__":
    main()
