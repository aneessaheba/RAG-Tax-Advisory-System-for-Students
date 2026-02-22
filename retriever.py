"""
Hybrid Retriever: combines vector search (ChromaDB) + keyword search (BM25)
Results are merged using Reciprocal Rank Fusion (RRF).

Why hybrid?
- Vector search finds semantically similar chunks (good for paraphrasing)
- BM25 finds exact keyword matches (good for form names, numbers, specific terms)
- RRF merges both rankings without needing to tune score weights
"""
import re
from rank_bm25 import BM25Okapi


def tokenize(text):
    """Lowercase and split into words, removing punctuation."""
    return re.findall(r'\b\w+\b', text.lower())


def reciprocal_rank_fusion(vector_ids, bm25_ids, k=60):
    """
    Merge two ranked lists using RRF.
    RRF score = sum of 1/(k + rank) across both lists.
    Higher score = better combined rank.
    k=60 is the standard default (dampens the impact of top ranks).
    """
    scores = {}
    for rank, chunk_id in enumerate(vector_ids):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    for rank, chunk_id in enumerate(bm25_ids):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever:
    """
    Loads all chunks from ChromaDB once at startup, builds a BM25 index,
    then for each query runs both vector and BM25 search and fuses results.
    """

    def __init__(self, collection):
        self.collection = collection

        print("  Loading all chunks for BM25 index...")
        all_data = collection.get(include=["documents", "metadatas"])
        self.all_ids = all_data["ids"]
        self.all_texts = all_data["documents"]
        self.all_metadatas = all_data["metadatas"]

        # Map chunk_id -> (text, metadata) for fast lookup
        self.chunk_map = {
            cid: {"text": text, "metadata": meta}
            for cid, text, meta in zip(self.all_ids, self.all_texts, self.all_metadatas)
        }

        # Build BM25 index over all chunk texts
        tokenized = [tokenize(t) for t in self.all_texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  BM25 index built over {len(self.all_ids)} chunks.")

    def retrieve(self, query, top_k=5, candidate_k=20):
        """
        Run hybrid retrieval for a query.

        1. Vector search: get top candidate_k chunks from ChromaDB
        2. BM25 search: score all chunks, take top candidate_k
        3. RRF: fuse both ranked lists, return top_k

        Returns list of dicts: [{text, metadata, chunk_id}, ...]
        Also returns the best vector similarity score (for confidence checking).
        """
        query_tokens = tokenize(query)

        # --- Vector search ---
        vec_results = self.collection.query(
            query_texts=[query],
            n_results=candidate_k,
            include=["distances"],
        )
        vector_ids = vec_results["ids"][0]
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: 1 - (distance/2) so range is 0-1
        best_vector_score = 1 - (vec_results["distances"][0][0] / 2) if vector_ids else 0.0

        # --- BM25 search ---
        bm25_scores = self.bm25.get_scores(query_tokens)
        # Get indices of top candidate_k BM25 scores
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:candidate_k]
        bm25_ids = [self.all_ids[i] for i in top_bm25_indices]

        # --- RRF fusion ---
        fused_ids = reciprocal_rank_fusion(vector_ids, bm25_ids)[:top_k]

        # --- Build result list ---
        results = []
        for chunk_id in fused_ids:
            if chunk_id in self.chunk_map:
                results.append({
                    "chunk_id": chunk_id,
                    "text": self.chunk_map[chunk_id]["text"],
                    "metadata": self.chunk_map[chunk_id]["metadata"],
                })

        return results, best_vector_score
