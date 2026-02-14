"""Step 5: Load embedded chunks into ChromaDB (free, no server needed)."""
import os
import json
import chromadb

BASE_DIR = os.path.dirname(__file__)
EMBEDDED_DIR = os.path.join(BASE_DIR, 'data_work', 'embedded_chunks')
CHROMA_DIR = os.path.join(BASE_DIR, 'data_work', 'chroma_db')

COLLECTION_NAME = "tax_docs"


def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    files = [f for f in os.listdir(EMBEDDED_DIR) if f.endswith('.json')]
    print(f"Found {len(files)} embedded chunks to load")

    # Process in batches of 100
    batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []

    for fname in files:
        with open(os.path.join(EMBEDDED_DIR, fname), 'r', encoding='utf-8') as f:
            chunk = json.load(f)

        batch_ids.append(chunk['chunk_id'])
        batch_embeddings.append(chunk['embedding'])
        batch_documents.append(chunk['text'])
        batch_metadatas.append({
            'doc_id': chunk.get('doc_id', ''),
            'source_type': chunk.get('source_type', ''),
            'title': chunk.get('title', ''),
            'year': chunk.get('year', ''),
            'country': chunk.get('country', ''),
            'page_number': chunk.get('page_number', 0),
        })

        if len(batch_ids) >= 100:
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )
            print(f"  Uploaded {len(batch_ids)} chunks")
            batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []

    # Upload remaining
    if batch_ids:
        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )
        print(f"  Uploaded {len(batch_ids)} chunks")

    print(f"Done. Total docs in collection: {collection.count()}")


if __name__ == "__main__":
    main()
