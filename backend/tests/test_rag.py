# backend/tests/test_rag.py
import pytest
from app.rag import get_embedding, retrieve_documents, generate_response


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
