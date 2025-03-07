# backend/app/database.py
import faiss
import numpy as np
from app.config import settings


class FAISSIndex:
    def __init__(self):
        self.index = None
        self.load_index()

    def load_index(self):
        try:
            self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
        except:
            self.index = faiss.IndexFlatL2(settings.DEFAULT_D_INDEX)

    def search(self, embedding: np.array, k=5):
        D, I = self.index.search(np.array([embedding]), k)
        return I[0]

    def add_embeddings(self, embeddings: np.array):
        self.index.add(embeddings)
        faiss.write_index(self.index, settings.FAISS_INDEX_PATH)


faiss_db = FAISSIndex()
