"""
Tax Advisor Bot for International Students
Uses RAG (ChromaDB + sentence-transformers) + Google Gemini free API

Retrieval: Hybrid (vector + BM25) with Reciprocal Rank Fusion
"""
import os
import json
from dotenv import load_dotenv
import chromadb
from google import genai
from retriever import HybridRetriever

load_dotenv()  # loads GEMINI_API_KEY from .env file

# --- Config ---
BASE_DIR = os.path.dirname(__file__)
CHROMA_DIR = os.path.join(BASE_DIR, 'tax_rag_data', 'data_work', 'chroma_db')
COLLECTION_NAME = "tax_docs"
TOP_K = 5


def get_student_info():
    """Ask the student a few questions to personalize advice."""
    print("\n=== International Student Tax Advisor ===\n")
    print("I'll ask a few questions to give you the right tax advice.\n")

    visa_type = input("1. What is your visa type? (F-1 / J-1 / M-1 / Other): ").strip() or "F-1"
    home_country = input("2. What is your home country?: ").strip() or "India"
    first_year = input("3. What year did you first enter the U.S.?: ").strip() or "2023"
    tax_year = input("4. What tax year are you filing for? (e.g. 2024): ").strip() or "2024"

    print("5. What types of income did you have? (comma-separated)")
    print("   Options: Wages/Salary, Scholarship/Fellowship, Stipend, On-campus job, OPT/CPT, None")
    income_raw = input("   Your answer: ").strip() or "None"
    income_types = [x.strip() for x in income_raw.split(",")]

    state = input("6. What U.S. state do you live in?: ").strip() or "CA"
    has_ssn = input("7. Do you have an SSN or ITIN? (yes/no): ").strip().lower()
    has_ssn = has_ssn in ("yes", "y")

    return {
        "visa_type": visa_type,
        "home_country": home_country,
        "first_entry_year": first_year,
        "tax_year": tax_year,
        "income_types": income_types,
        "state": state,
        "has_ssn_or_itin": has_ssn,
    }


def build_query(student_info, question):
    """Enrich the query with student profile context for better retrieval."""
    parts = [
        question,
        f"{student_info['visa_type']} student",
        f"from {student_info['home_country']}",
        f"tax year {student_info['tax_year']}",
    ]
    if any("OPT" in t or "CPT" in t for t in student_info['income_types']):
        parts.append("OPT CPT employment")
    return " ".join(parts)


def format_context(chunks):
    """Format retrieved chunks into a readable context string."""
    parts = []
    for c in chunks:
        meta = c["metadata"]
        source = f"[{meta.get('title', 'Unknown')} - p.{meta.get('page_number', '?')}]"
        parts.append(f"{source}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def ask_gemini(student_info, context, question):
    """Send the question + context to Gemini."""
    prompt = f"""You are a helpful tax advisor for international students in the U.S.

Student profile:
- Visa: {student_info['visa_type']}
- Home country: {student_info['home_country']}
- First U.S. entry: {student_info['first_entry_year']}
- Tax year: {student_info['tax_year']}
- Income types: {', '.join(student_info['income_types'])}
- State: {student_info['state']}
- Has SSN/ITIN: {'Yes' if student_info['has_ssn_or_itin'] else 'No'}

Use ONLY the provided reference documents to answer. If the documents don't cover something,
say so clearly. Always remind the student this is general guidance, not professional tax advice.

--- REFERENCE DOCUMENTS ---
{context}
--- END DOCUMENTS ---

Student's question: {question}

Provide a clear, helpful answer:"""

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY in .env")
        print("Get a free key at: https://aistudio.google.com/apikey")
        return

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    print("Building hybrid retriever (vector + BM25)...")
    retriever = HybridRetriever(collection)

    student_info = get_student_info()

    with open(os.path.join(BASE_DIR, "user_profile.json"), 'w') as f:
        json.dump(student_info, f, indent=2)

    print("\n=== Ready! Ask your tax questions (type 'quit' to exit) ===\n")

    while True:
        question = input("\nYour question: ").strip()
        if not question or question.lower() in ('quit', 'exit', 'q'):
            print("Goodbye! Remember: consult a tax professional for specific advice.")
            break

        query = build_query(student_info, question)
        print("Searching tax documents (hybrid)...")
        chunks, confidence = retriever.retrieve(query, top_k=TOP_K)
        context = format_context(chunks)

        print("Generating answer...\n")
        answer = ask_gemini(student_info, context, question)
        print(f"\n{answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
