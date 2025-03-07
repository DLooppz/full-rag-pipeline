from backend.app.rag import retrieve_documents

print(retrieve_documents("How does FAISS indexing work?", k=3))
