# backend/app/config.py
from pydantic import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    FAISS_INDEX_PATH: str = "data/faiss_index"
    DEFAULT_D_INDEX: int = 768

    class Config:
        env_file = ".env"


settings = Settings()
