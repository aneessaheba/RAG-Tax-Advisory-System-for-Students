"""
RAG Evaluation Script
Measures 4 metrics for each test question:
  1. Context Relevance  - how well retrieved chunks match the query
  2. Hit Rate           - did any chunk contain the expected answer keywords
  3. Answer Relevance   - how well the generated answer addresses the question
  4. Faithfulness       - how grounded the answer is in the retrieved context
All metrics are 0.0 to 1.0 (higher is better).
"""
import os
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
CHROMA_DIR = os.path.join(BASE_DIR, 'tax_rag_data', 'data_work', 'chroma_db')
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, 'ground_truth.json')
RESULTS_PATH = os.path.join(BASE_DIR, 'evaluation_results.json')

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "tax_docs"
TOP_K = 5


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def context_relevance(query_embedding, chunk_embeddings):
    """Average cosine similarity between the query and each retrieved chunk."""
    scores = [cosine_similarity(query_embedding, ce) for ce in chunk_embeddings]
    return round(float(np.mean(scores)), 4)


def hit_rate(chunk_texts, expected_keywords):
    """1.0 if any chunk contains all expected keywords (case-insensitive), else 0.0."""
    joined = " ".join(chunk_texts).lower()
    hit = all(kw.lower() in joined for kw in expected_keywords)
    return 1.0 if hit else 0.0


def answer_relevance(question_embedding, answer_embedding):
    """Cosine similarity between the question and the generated answer."""
    return round(cosine_similarity(question_embedding, answer_embedding), 4)


def faithfulness(answer_embedding, chunk_embeddings):
    """Cosine similarity between the answer and the average context embedding."""
    avg_context = np.mean(chunk_embeddings, axis=0)
    return round(cosine_similarity(answer_embedding, avg_context), 4)


def generate_answer(question, chunks, api_key, model_name="gemini-2.0-flash"):
    """Call Gemini to generate an answer grounded in the retrieved chunks."""
    context = "\n\n".join(chunks)
    prompt = f"""You are a tax advisor for international students.
Answer the question using ONLY the context below. Be concise.

Context:
{context}

Question: {question}
Answer:"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text.strip()


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY in .env")
        return

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    with open(GROUND_TRUTH_PATH, 'r') as f:
        test_cases = json.load(f)

    results = []
    totals = {"context_relevance": 0, "hit_rate": 0, "answer_relevance": 0, "faithfulness": 0}

    print(f"\nRunning evaluation on {len(test_cases)} questions...\n")
    print(f"{'#':<4} {'Question':<55} {'CtxRel':>7} {'Hit':>5} {'AnsRel':>7} {'Faith':>7}")
    print("-" * 90)

    for i, tc in enumerate(test_cases):
        question = tc["question"]
        expected_keywords = tc["expected_keywords"]

        # Embed the question
        q_embedding = embed_model.encode(question).tolist()

        # Retrieve top K chunks
        retrieved = collection.query(
            query_embeddings=[q_embedding],
            n_results=TOP_K,
            include=["documents", "embeddings"],
        )
        chunk_texts = retrieved["documents"][0]
        chunk_embeddings = retrieved["embeddings"][0]

        # Generate answer
        answer = generate_answer(question, chunk_texts, api_key)
        answer_embedding = embed_model.encode(answer).tolist()

        # Compute metrics
        m_context_relevance = context_relevance(q_embedding, chunk_embeddings)
        m_hit_rate = hit_rate(chunk_texts, expected_keywords)
        m_answer_relevance = answer_relevance(q_embedding, answer_embedding)
        m_faithfulness = faithfulness(answer_embedding, chunk_embeddings)

        # Store result
        result = {
            "question": question,
            "expected_keywords": expected_keywords,
            "answer": answer,
            "metrics": {
                "context_relevance": m_context_relevance,
                "hit_rate": m_hit_rate,
                "answer_relevance": m_answer_relevance,
                "faithfulness": m_faithfulness,
            }
        }
        results.append(result)

        for k in totals:
            totals[k] += result["metrics"][k]

        # Print row
        short_q = question[:53] + ".." if len(question) > 53 else question
        print(f"{i+1:<4} {short_q:<55} {m_context_relevance:>7.3f} {int(m_hit_rate):>5} {m_answer_relevance:>7.3f} {m_faithfulness:>7.3f}")

    # Averages
    n = len(test_cases)
    averages = {k: round(v / n, 4) for k, v in totals.items()}

    print("-" * 90)
    print(f"{'AVERAGE':<59} {averages['context_relevance']:>7.3f} {averages['hit_rate']:>5.2f} {averages['answer_relevance']:>7.3f} {averages['faithfulness']:>7.3f}")

    # Save results
    output = {
        "model": EMBED_MODEL,
        "llm": "gemini-2.0-flash",
        "top_k": TOP_K,
        "num_questions": n,
        "averages": averages,
        "per_question": results,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")
    print("\nSummary:")
    print(f"  Context Relevance : {averages['context_relevance']:.3f}  (are retrieved chunks related to the question?)")
    print(f"  Hit Rate          : {averages['hit_rate']:.2f}   (did we find chunks with the right answer?)")
    print(f"  Answer Relevance  : {averages['answer_relevance']:.3f}  (does the answer address the question?)")
    print(f"  Faithfulness      : {averages['faithfulness']:.3f}  (is the answer grounded in retrieved context?)")


if __name__ == "__main__":
    main()
