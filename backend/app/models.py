from pydantic_settings import BaseSettings


class QueryRequest(BaseSettings):
    question: str
