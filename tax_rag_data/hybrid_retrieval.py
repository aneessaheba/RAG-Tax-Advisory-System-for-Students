import json
def load_user_profile(path='user_profile.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
import os
from elasticsearch import Elasticsearch
import numpy as np

INDEX_NAME = 'tax_docs_hybrid'
ES_HOST = os.environ.get('ES_HOST', 'http://localhost:9200')
ES_USER = os.environ.get('ES_USER')
ES_PASS = os.environ.get('ES_PASS')

# --- Connect to Elasticsearch ---
def get_es_client():
    if ES_USER and ES_PASS:
        return Elasticsearch(ES_HOST, basic_auth=(ES_USER, ES_PASS))
    return Elasticsearch(ES_HOST)

# --- Hybrid retrieval function ---
def hybrid_retrieve(user_profile, query, query_embedding, top_k=5):
    """
    user_profile: dict with keys visa_type, tax_year, first_entry_year, home_country, income_types, state
    query: str (user's question)
    query_embedding: list[float] (embedding vector for enriched query)
    top_k: int (number of results)
    """
    es = get_es_client()
    # Enrich query with user profile
    # Boosting rules:
    # - country == home_country: +0.5 boost (treaty relevance)
    # - text contains visa_type: +0.3 boost (student/visa-specific sections)
    # - text contains any income_type: +0.2 per match (income-specific sections)
    enriched_query = f"{query} Visa: {user_profile.get('visa_type','')} Year: {user_profile.get('tax_year','')} FirstEntry: {user_profile.get('first_entry_year','')} Country: {user_profile.get('home_country','')} Income: {' '.join(user_profile.get('income_types', []))} State: {user_profile.get('state','')}"
    # Build ES query
    es_query = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"text": enriched_query}},
                            {"term": {"country": user_profile.get('home_country','')}},
                            {"match": {"text": user_profile.get('visa_type','')}},
                            {"terms": {"text": user_profile.get('income_types', [])}}
                        ],
                        "minimum_should_match": 1
                    }
                },
                "script": {
                    "source": """
                        double bm25 = _score;
                        double dense = cosineSimilarity(params.query_vector, 'embedding');
                        double boost = 1.0;
                        if (doc['country'].size()!=0 && doc['country'].value == params.home_country) { boost += 0.5; }
                        if (doc['text'].size()!=0 && doc['text'].value.contains(params.visa_type)) { boost += 0.3; }
                        for (t in params.income_types) {
                          if (doc['text'].size()!=0 && doc['text'].value.contains(t)) { boost += 0.2; }
                        }
                        return bm25 + dense * boost;
                    """,
                    "params": {
                        "query_vector": query_embedding,
                        "home_country": user_profile.get('home_country',''),
                        "visa_type": user_profile.get('visa_type',''),
                        "income_types": user_profile.get('income_types', [])
                    }
                }
            }
        }
    }
    res = es.search(index=INDEX_NAME, body=es_query)
    return [hit['_source'] for hit in res['hits']['hits']]

# --- Example usage ---
if __name__ == "__main__":
    def test_boosting():
        """
        Test retrieval with different user profiles to validate boosting logic.
        """
        import numpy as np
        user_profiles = [
            # Treaty boost
            {'visa_type': 'F-1', 'tax_year': '2025', 'first_entry_year': '2023', 'home_country': 'India', 'income_types': ['wage'], 'state': 'CA'},
            # Visa boost
            {'visa_type': 'J-1', 'tax_year': '2025', 'first_entry_year': '2023', 'home_country': 'USA', 'income_types': ['scholarship'], 'state': 'NY'},
            # Income boost
            {'visa_type': 'B-2', 'tax_year': '2025', 'first_entry_year': '2023', 'home_country': 'Canada', 'income_types': ['scholarship'], 'state': 'TX'},
        ]
        query = "What tax treaty benefits apply?"
        for i, profile in enumerate(user_profiles, 1):
            print(f"\nTest {i}: {profile}")
            query_embedding = np.random.rand(384).tolist()  # Replace with real embedding
            results = hybrid_retrieve(profile, query, query_embedding)
            for j, doc in enumerate(results, 1):
                print(f"  Result {j}: Doc ID: {doc['doc_id']} | Chunk ID: {doc['chunk_id']} | Text: {doc['text'][:100]}...")

    # Load user profile from file
    user_profile = load_user_profile()
    query = input("Enter your tax question: ").strip()
    # You must generate the embedding for the enriched query using your embedding model
    # For demo, use a random vector (replace with real embedding in production)
    query_embedding = np.random.rand(384).tolist()  # Replace with real embedding
    results = hybrid_retrieve(user_profile, query, query_embedding)
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:\nDoc ID: {doc['doc_id']}\nChunk ID: {doc['chunk_id']}\nText: {doc['text'][:200]}...\n")
