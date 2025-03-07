# backend/app/main.py
from fastapi import FastAPI
from app.rag import generate_response
from app.models import QueryRequest

app = FastAPI()


@app.get("/")
def home():
    return {"message": "RAG Pipeline is running"}


@app.post("/query/")
def query_rag(request: QueryRequest):
    response = generate_response(request.question)
    return {"answer": response}
