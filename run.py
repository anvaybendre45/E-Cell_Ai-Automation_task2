import uvicorn

if __name__ == "__main__":
    # This launches Uvicorn while keeping C:\rag_project as the core directory path
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
