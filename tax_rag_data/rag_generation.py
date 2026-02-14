import json
from typing import List

def build_rag_prompt(question: str, retrieved_chunks: List[dict], user_profile: dict) -> str:
    """
    Build a prompt for RAG generation with citations and profile context.
    """
    context = "\n\n".join([
        f"[doc_id: {c['doc_id']}, page: {c.get('page_number', '?')}] {c['text']}" for c in retrieved_chunks
    ])
    profile_str = f"Visa: {user_profile.get('visa_type','')}, Year: {user_profile.get('tax_year','')}, First Entry: {user_profile.get('first_entry_year','')}, Country: {user_profile.get('home_country','')}, Income Types: {', '.join(user_profile.get('income_types', []))}, State: {user_profile.get('state','')}"
    prompt = f"""
You are a tax law assistant. Answer the user's question using ONLY the provided context. Always cite sources using [doc_id, page] after each claim. If information is missing, ask a clarifying question. If you cannot answer from the context, say so and do not speculate.

User profile: {profile_str}

Question: {question}

Context:
{context}

Answer (with citations):
"""
    return prompt

# Example usage
def main():
    # Load user profile and retrieved chunks (replace with real data in production)
    with open('user_profile.json', 'r', encoding='utf-8') as f:
        user_profile = json.load(f)
    # Simulate retrieval
    retrieved_chunks = [
        {"doc_id": "irs_pub519_2025", "page_number": 3, "text": "Nonresident aliens must file Form 8843 if they are present in the U.S. under F-1 status."},
        {"doc_id": "irs_pub519_2025", "page_number": 5, "text": "Form 8843 is required even if you have no income."}
    ]
    question = "Do I need to file form 8843 as an F-1 student with no income?"
    prompt = build_rag_prompt(question, retrieved_chunks, user_profile)
    print(prompt)

if __name__ == "__main__":
    main()
