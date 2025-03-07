# backend/app/config.py
from pydantic import v1 as pydantic_v1


class Settings(pydantic_v1.BaseSettings):
    OPENAI_API_KEY: str
    FAISS_INDEX_PATH: str = "data/faiss_index.index"
    FAISS_TEST_INDEX_PATH: str = "data/test_faiss_index.index"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    DEFAULT_D_INDEX: int = 1536

    class Config:
        env_file = "backend/.env"


settings = Settings()
