# src/train.py
import os
from google import genai
from google.genai import types
from src.preprocess import DocumentProcessor
from src.features import HybridSearchEngine

class RAGPipeline:
    def __init__(self):
        """
        Initializes Stage 3 context orchestration and links to the 
        Google Gemini API developer tier.
        """
        # Fetching your secure API token from your system environment variables
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("CRITICAL ERROR: GEMINI_API_KEY environment variable is not set.")
        
        self.client = genai.Client(api_key=api_key)
        self.processor = DocumentProcessor()
        self.search_engine = HybridSearchEngine()

    def initialize_pipeline(self, data_dir: str):
        """
        Runs the full pipeline end-to-end: loads multi-format chunks, runs dense/sparse 
        indexing, and serializes the local vector store.
        """
        print("\n--- STARTING PIPELINE INITIALIZATION ---")
        print("[1/2] Preprocessing raw text documentation segments...")
        chunks = self.processor.process_directory(data_dir)
        print(f"[STAGE 1 SUCCESS] Extracted and cleaned {len(chunks)} text segments.")
        
        print(f"[2/2] Running dense transformers and populating indices...")
        self.search_engine.index_documents(chunks)
        print("[STAGE 2 SUCCESS] Pipeline architecture successfully established and saved to disk.")

    def query(self, user_prompt: str) -> dict:
        """
        Executes search queries, builds anti-hallucination guardrails, 
        and orchestrates the LLM inference.
        """
        print(f"\n[QUERY RECEIVED] Processing question: '{user_prompt}'")
        
        # 1. Fetch top hybrid search results from our database chunks
        retrieved_contexts = self.search_engine.hybrid_retrieve(user_prompt, top_k=3)
        print(f"[RETRIEVAL] Pulled top {len(retrieved_contexts)} factual context frames.")
        
        # 2. Compile the matched pieces into a clean reference string
        context_str = "\n---\n".join([
            f"Source File: {c['metadata']['source']} (Page {c['metadata']['page']})\nContent: {c['text']}"
            for c in retrieved_contexts
        ])
        
        # 3. Apply strict anti-hallucination system guardrails 
        system_instruction = (
            "You are a factual operational assistant. Answer the user query using only the provided context "
            "segments below. If the documentation does not contain answers to the query, explicitly state: "
            "'I cannot determine the answer based on provided operational documentation.' Do not invent facts."
        )
        
        user_content = f"Context Material:\n{context_str}\n\nQuery: {user_prompt}"
        
        print("[LLM INFERENCE] Routing context packets to gemini-2.5-flash...")
        
        # 4. Request generation from gemini-2.5-flash using the zero-cost developer tier
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1 # Forces deterministic accuracy over creativity to eliminate hallucinations
            )
        )
        
        return {
            "answer": response.text,
            "retrieved_context": retrieved_contexts
        }

if __name__ == "__main__":
    print("Testing Stage 3 Pipeline Orchestration Loop...")
    try:
        pipeline = RAGPipeline()
        pipeline.initialize_pipeline("data")
        
        # Let's run a live semantic query test using your dataset!
        # Modify this prompt based on what kind of document you dropped inside data/
        test_prompt = "What is the primary objective or operational task described in the documentation?"
        
        result = pipeline.query(test_prompt)
        
        print("\n=================== CHOSEN CONTEXT SOURCES ===================")
        for idx, src in enumerate(result["retrieved_context"], start=1):
            print(f"Match #{idx}: {src['metadata']['source']} (Page {src['metadata']['page']}) [Route: {src['origin']}]")
            
        print("\n=================== GENERATED RESPONSE ===================")
        print(result["answer"])
        print("==========================================================")
        
    except Exception as e:
        print(f"\n[PIPELINE EXCEPTION] Execution paused: {e}")