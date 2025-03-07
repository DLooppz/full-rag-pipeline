# backend/app/rag.py
import openai
import numpy as np
from app.database import faiss_db
from app.config import settings


def get_embedding(text: str):
    response = openai.embeddings.create(input=text, model=settings.EMBEDDING_MODEL)
    return np.array(response.data[0].embedding, dtype=np.float32)


def retrieve_documents(query: str, k=5):
    query_embedding = get_embedding(query)
    doc_indices = faiss_db.search(query_embedding, k)
    return doc_indices


def generate_response(question: str):
    retrieved_docs = retrieve_documents(question)
    prompt = f"Based on these documents: {retrieved_docs}, answer: {question}"
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
