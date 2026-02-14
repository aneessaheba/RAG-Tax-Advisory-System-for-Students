import os
import json
import re
from itertools import count

CLEAN_DIR = os.path.join(os.path.dirname(__file__), 'data_work', 'clean_docs')
CHUNK_DIR = os.path.join(os.path.dirname(__file__), 'data_work', 'chunks')
os.makedirs(CHUNK_DIR, exist_ok=True)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def split_into_chunks(text, chunk_size, overlap):
    words = re.findall(r'\S+', text)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(' '.join(chunk_words))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

def process_file(filepath, chunk_id_counter):
    with open(filepath, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    doc_id = doc.get('doc_id')
    source_type = doc.get('source_type')
    title = doc.get('title')
    year = doc.get('year')
    country = doc.get('country')
    for page in doc.get('pages', []):
        page_number = page.get('page_number')
        text = page.get('text', '')
        chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            chunk_id = f"{doc_id}_p{page_number}_c{next(chunk_id_counter)}"
            chunk = {
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'source_type': source_type,
                'title': title,
                'year': year,
                'country': country,
                'page_number': page_number,
                'text': chunk_text
            }
            out_path = os.path.join(CHUNK_DIR, f"{chunk_id}.json")
            with open(out_path, 'w', encoding='utf-8') as out_f:
                json.dump(chunk, out_f, ensure_ascii=False, indent=2)
            print(f"Saved chunk: {out_path}")

def main():
    chunk_id_counter = count(1)
    for fname in os.listdir(CLEAN_DIR):
        if not fname.endswith('.json'):
            continue
        filepath = os.path.join(CLEAN_DIR, fname)
        try:
            process_file(filepath, chunk_id_counter)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    main()
