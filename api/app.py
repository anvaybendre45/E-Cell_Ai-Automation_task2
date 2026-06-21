# api/app.py
import os
import sys

# Crucial path patcher must run before any internal modules are imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.train import RAGPipeline

app = FastAPI(title="RAG Chatbot API Gateway")
pipeline = None

class QueryRequest(BaseModel):
    prompt: str

@app.on_event("startup")
def startup_event():
    global pipeline
    try:
        pipeline = RAGPipeline()
        pipeline.initialize_pipeline("data")
    except Exception as e:
        print(f"Startup error: {e}")

@app.post("/query")
def run_query(request: QueryRequest):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not ready.")
    try:
        result = pipeline.query(request.prompt)
        return {"answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Listening on 0.0.0.0 enables local programs like n8n to find it instantly
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)