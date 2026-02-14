import os
import json
import re

PARSED_DIR = os.path.join(os.path.dirname(__file__), 'data_work', 'parsed_docs')
CLEAN_DIR = os.path.join(os.path.dirname(__file__), 'data_work', 'clean_docs')
os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_text(text):
    # Fix hyphenated word breaks across lines (e.g., "exam-\nple" -> "example")
    text = re.sub(r'([a-zA-Z])-\n([a-zA-Z])', r'\1\2', text)
    # Merge broken sentence line breaks (e.g., "sentence.\nNext" -> "sentence. Next")
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Remove excessive blank lines (more than 2 in a row)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Normalize spacing (remove trailing, leading, and excessive spaces)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = text.strip()
    return text

def clean_document(doc):
    cleaned = doc.copy()
    cleaned['pages'] = []
    for page in doc.get('pages', []):
        cleaned_page = {
            'page_number': page['page_number'],
            'text': clean_text(page['text'])
        }
        cleaned['pages'].append(cleaned_page)
    return cleaned

def main():
    for fname in os.listdir(PARSED_DIR):
        if not fname.endswith('.json'):
            continue
        in_path = os.path.join(PARSED_DIR, fname)
        out_path = os.path.join(CLEAN_DIR, fname)
        try:
            with open(in_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            cleaned = clean_document(doc)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned, f, ensure_ascii=False, indent=2)
            print(f"Cleaned: {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    main()
