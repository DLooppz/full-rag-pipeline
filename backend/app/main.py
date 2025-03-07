# backend/app/main.py
from fastapi import FastAPI
from app.rag import generate_response

app = FastAPI()


@app.get("/")
def home():
    return {"message": "RAG Pipeline is running"}


@app.post("/query/")
def query_rag(question: str):
    response = generate_response(question)
    return {"answer": response}
