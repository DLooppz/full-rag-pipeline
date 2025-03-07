import os
import faiss
import numpy as np
from app.config import settings


class FAISSIndex:
    def __init__(self, index_path=settings.FAISS_INDEX_PATH):
        self.index_path = index_path
        self.index = None
        self.dimension = 1536  # Ensure this matches OpenAI embeddings
        self.load_index()

    def load_index(self):
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)

        if os.path.exists(settings.FAISS_INDEX_PATH):
            try:
                self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
                if self.index.d != self.dimension:
                    raise ValueError(
                        f"FAISS index dimension {self.index.d} does not match expected {self.dimension}"
                    )
            except:
                print("Failed to load FAISS index, creating a new one.")
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            print("No FAISS index found, creating a new one.")
            self.index = faiss.IndexFlatL2(self.dimension)

    def search(self, embedding: np.array, k=5):
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(np.array([embedding]), k)
        return I[0]

    def add_embeddings(self, embeddings: np.array):
        self.index.add(embeddings)
        faiss.write_index(self.index, settings.FAISS_INDEX_PATH)
        print(f"FAISS index saved with {self.index.ntotal} embeddings.")


faiss_db = FAISSIndex()
