# src/features.py
print("[DIAGNOSTIC] Starting features.py script execution...")

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import numpy as np

print("[DIAGNOSTIC] All core libraries imported successfully.")

class HybridSearchEngine:
    def __init__(self, db_path: str = "data/chroma_db", collection_name: str = "tech_docs"):
        """
        Initializes Stage 2: Local persistent vector storage configuration
        alongside a sparse keyword fallback registry.
        """
        print(f"[DIAGNOSTIC] Initializing ChromaDB client at: {db_path}")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        print("[DIAGNOSTIC] Setting up local dense transformer function (all-MiniLM-L6-v2)...")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25 = None
        self.corpus_chunks = []

    def index_documents(self, chunked_docs: list[dict]):
        """Converts text blocks into dense math matrices and saves to local disk."""
        if not chunked_docs:
            print("[WARNING] Indexing aborted: Provided text chunks corpus is empty.")
            return
        
        self.corpus_chunks = chunked_docs
        ids = [f"id_{i}" for i in range(len(chunked_docs))]
        documents = [doc["text"] for doc in chunked_docs]
        metadatas = [doc["metadata"] for doc in chunked_docs]
        
        print(f"[STAGE 2 LOG] Generating embeddings for {len(documents)} text blocks...")
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        
        print("[STAGE 2 LOG] Building sparse token keyword-matching matrices...")
        tokenized_corpus = [doc.lower().split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("[STAGE 2 LOG] Storage indices successfully synchronized and saved.")

    def hybrid_retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Executes hybrid search query routing."""
        if not self.bm25:
            return []
        dense_results = self.collection.query(query_texts=[query], n_results=top_k)
        dense_docs = dense_results['documents'][0] if dense_results['documents'] else []
        dense_meta = dense_results['metadatas'][0] if dense_results['metadatas'] else []
        
        tokenized_query = query.lower().split(" ")
        sparse_scores = self.bm25.get_scores(tokenized_query)
        top_sparse_indices = np.argsort(sparse_scores)[::-1][:top_k]
        
        combined_results = []
        for text, meta in zip(dense_docs, dense_meta):
            combined_results.append({"text": text, "metadata": meta, "origin": "dense_vector"})
            
        for idx in top_sparse_indices:
            sparse_text = self.corpus_chunks[idx]["text"]
            if not any(item["text"] == sparse_text for item in combined_results):
                combined_results.append({
                    "text": sparse_text, 
                    "metadata": self.corpus_chunks[idx]["metadata"],
                    "origin": "sparse_keyword"
                })
        return combined_results[:top_k]

# THIS BLOCK FORCES STANDALONE EXECUTION OUTPUTS
if __name__ == "__main__":
    print("\n--- STANDALONE BLOCK TRIGGERED ---")
    print("Testing Stage 2 Hybrid Search structures locally...")
    engine = HybridSearchEngine()
    print("Hybrid Search Engine structures initialized smoothly.")
    print("-----------------------------------\n")