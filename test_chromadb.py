#!/usr/bin/env python3
"""Test script to check ChromaDB API capabilities"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("=" * 60)
print("Testing ChromaDB API Capabilities")
print("=" * 60)

# Initialize embeddings and vector DB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"device": "mps"}
)

vector_db = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)

# Test question
test_question = "하이브는 어떤 회사인가?"

print(f"\nTest Question: {test_question}")
print("-" * 60)

# Check available methods
print("\n--- Checking Available Methods ---")
methods = [
    'similarity_search_with_scores',
    'similarity_search',
    'query',
    '_collection',
    'get_relevant_documents'
]

for method in methods:
    has_method = hasattr(vector_db, method)
    print(f"✓ {method}: {has_method}")

# Method 1: similarity_search_with_scores
print("\n--- Method 1: similarity_search_with_scores() ---")
try:
    results = vector_db.similarity_search_with_scores(test_question, k=3)
    print(f"✓ Works! Got {len(results)} results")
    for i, (doc, score) in enumerate(results, 1):
        print(f"  {i}. Score: {score:.4f}, Content: {doc.page_content[:80]}...")
except AttributeError as e:
    print(f"✗ Method not found: {e}")
except Exception as e:
    print(f"✗ Error: {e}")

# Method 2: _collection.query
print("\n--- Method 2: _collection.query() ---")
try:
    collection = vector_db._collection
    print(f"✓ Collection object exists")
    
    # Get embedding for question
    question_embedding = embeddings.embed_query(test_question)
    print(f"✓ Question embedding generated: {len(question_embedding)} dimensions")
    
    # Try query
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )
    print(f"✓ Query executed!")
    print(f"  Results keys: {results.keys()}")
    if 'distances' in results and results['distances']:
        distances = results['distances'][0]
        print(f"  Distances: {distances}")
        similarities = [1 / (1 + d) for d in distances]
        print(f"  Converted Similarities: {similarities}")
except AttributeError as e:
    print(f"✗ Attribute error: {e}")
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")

# Method 3: similarity_search (basic)
print("\n--- Method 3: similarity_search() ---")
try:
    results = vector_db.similarity_search(test_question, k=3)
    print(f"✓ Works! Got {len(results)} results")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. Content: {doc.page_content[:80]}...")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
