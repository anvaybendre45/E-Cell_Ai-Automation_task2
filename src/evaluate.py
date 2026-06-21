import os
import re
from google import genai
from google.genai import types

class RAGEvaluator:
    def __init__(self):
        """Initializes the LLM-as-a-Judge evaluation tier."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("CRITICAL ERROR: GEMINI_API_KEY is not configured for Evaluation.")
        self.client = genai.Client(api_key=api_key)

    def _parse_score(self, response_text: str) -> float:
        """Helper to extract a clean numerical float score from LLM text analysis."""
        match = re.search(r"score:\s*([0-1]\.\d+|[01])", response_text.lower())
        if match:
            return float(match.group(1))
        return 0.5 # Safe mid-tier fallback if structural formatting fails

    def evaluate_groundedness(self, answer: str, context: str) -> float:
        """Evaluates if the answer stays 100% strictly bounded within the context."""
        prompt = (
            f"You are an expert quality assurance judge.\n"
            f"Analyze the provided ANSWER and check if every claim within it is explicitly supported by the CONTEXT.\n"
            f"If the answer invents facts or adds outside info, penalize the score.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"Output your assessment exactly in this format: 'SCORE: [float between 0.0 and 1.0]'"
        )
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0) # Forces rigorous rule tracking
            )
            return self._parse_score(response.text)
        except Exception as e:
            print(f"Groundedness evaluation error: {e}")
            return 0.0

    def evaluate_context_relevance(self, query: str, context: str) -> float:
        """Evaluates if the hybrid search engine pulled chunks directly relevant to the user's prompt."""
        prompt = (
            f"You are an expert data retrieval auditor.\n"
            f"Evaluate if the retrieved CONTEXT blocks contain the direct information required to answer the user's QUERY.\n"
            f"If the context is irrelevant fluff, score it 0.0. If it perfectly addresses the query, score it 1.0.\n\n"
            f"QUERY:\n{query}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"Output your assessment exactly in this format: 'SCORE: [float between 0.0 and 1.0]'"
        )
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            return self._parse_score(response.text)
        except Exception as e:
            print(f"Context relevance evaluation error: {e}")
            return 0.0

if __name__ == "__main__":
    print("Testing Stage 4 Evaluator structures standalone...")
    evaluator = RAGEvaluator()
    print("Evaluator structures compiled cleanly.")
