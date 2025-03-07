import pytest
import json
import numpy as np
from backend.app.rag import get_embedding, retrieve_documents, generate_response
from backend.app.database import FAISSIndex
from backend.app.config import settings

# Use a separate test FAISS index
faiss_db = FAISSIndex(index_path=settings.FAISS_TEST_INDEX_PATH)

print(f"Using FAISS test index at: {settings.FAISS_TEST_INDEX_PATH}")

# Load test documents
with open("backend/tests/test_data/test_documents.json", "r") as f:
    TEST_DOCUMENTS = json.load(f)


# Ensure FAISS index is cleared before tests
def setup_module(module):
    """Clear FAISS index before running tests and add test embeddings."""
    print(f"Clearing FAISS index at {settings.FAISS_TEST_INDEX_PATH}")
    embeddings = np.array(
        [get_embedding(doc) for doc in TEST_DOCUMENTS], dtype=np.float32
    )
    faiss_db.index.reset()  # Clear previous FAISS index
    faiss_db.add_embeddings(embeddings)
    print(f"Test FAISS index now contains {faiss_db.index.ntotal} vectors")


def test_embedding():
    text = "What is FAISS?"
    embedding = get_embedding(text)
    assert len(embedding) > 0, "Embedding should not be empty"


def test_retrieve_documents():
    query = "How does FAISS indexing work?"
    results = retrieve_documents(query, k=3)
    assert len(results) == 3, "Should retrieve 3 documents"


def test_generate_response():
    question = "Explain FAISS in simple terms"
    response = generate_response(question)
    assert (
        isinstance(response, str) and len(response) > 0
    ), "Response should be a non-empty string"
