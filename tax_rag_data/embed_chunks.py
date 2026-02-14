"""Step 4: Embed text chunks using free sentence-transformers model."""
import os
import json
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(__file__)
CHUNK_DIR = os.path.join(BASE_DIR, 'data_work', 'chunks')
EMBEDDED_DIR = os.path.join(BASE_DIR, 'data_work', 'embedded_chunks')
os.makedirs(EMBEDDED_DIR, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"  # Free, fast, 384-dim embeddings


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    files = [f for f in os.listdir(CHUNK_DIR) if f.endswith('.json')]
    print(f"Found {len(files)} chunks to embed")

    for i, fname in enumerate(files):
        out_path = os.path.join(EMBEDDED_DIR, fname)
        if os.path.exists(out_path):
            continue  # skip already embedded

        in_path = os.path.join(CHUNK_DIR, fname)
        with open(in_path, 'r', encoding='utf-8') as f:
            chunk = json.load(f)

        text = chunk.get('text', '').strip()
        if not text:
            continue

        embedding = model.encode(text).tolist()
        chunk['embedding'] = embedding

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

        if (i + 1) % 50 == 0:
            print(f"  Embedded {i + 1}/{len(files)}")

    print("Done embedding all chunks.")


if __name__ == "__main__":
    main()
