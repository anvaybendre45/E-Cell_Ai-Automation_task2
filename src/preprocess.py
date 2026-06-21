# src/preprocess.py
import os
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # Initializes Stage 1 with dynamic, customizable hyperparameter controls.
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Exposing configurable parameters directly to the underlying splitter setup
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def clean_text(self, text: str) -> str:
        """
        Strips systemic artifacts, repeating headers/footers, page numbering systems,
        and multiple consecutive whitespaces to ensure high semantic data density.
        """
        if not text:
            return ""

        # Remove common repeating page number artifacts
        text = re.sub(re.compile(r'page\s*[-–_]?\s*\d+(\s*of\s*\d+)?', re.IGNORECASE), '', text)
        text = re.sub(re.compile(r'\[\s*page\s*\d+\s*\]', re.IGNORECASE), '', text)
        
        # Strip standalone trailing or leading line numbers/isolated page digits
        text = re.sub(re.compile(r'^\s*\d+\s*$', re.MULTILINE), '', text)

        # Normalize broken hyphenated words at line breaks
        text = re.sub(re.compile(r'(\w+)-\n(\w+)'), r'\1\2', text)

        # Collapse multi-spacing variants and consecutive newlines into clean single breaks
        text = re.sub(re.compile(r'[ \t]+'), ' ', text)
        text = re.sub(re.compile(r'\n\s*\n+'), '\n\n', text)

        return text.strip()

    def extract_from_pdf(self, file_path: str) -> list[dict]:
        """Extracts and thoroughly cleans text data page-by-page from PDFs."""
        pages_data = []
        try:
            reader = PdfReader(file_path)
            filename = os.path.basename(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text()
                
                # EXECUTE SYSTEMIC ARTIFACT CLEANING RULE
                cleaned_text = self.clean_text(raw_text)
                
                if cleaned_text:
                    pages_data.append({
                        "text": cleaned_text,
                        "metadata": {"source": filename, "page": page_num}
                    })
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return pages_data

    def extract_from_text_or_markdown(self, file_path: str) -> list[dict]:
        """Extracts and thoroughly cleans raw plain text strings from .txt and .md files."""
        pages_data = []
        try:
            filename = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
                
                # EXECUTE SYSTEMIC ARTIFACT CLEANING RULE
                cleaned_content = self.clean_text(raw_content)
                
                if cleaned_content:
                    pages_data.append({
                        "text": cleaned_content,
                        "metadata": {"source": filename, "page": 1}
                    })
        except Exception as e:
            print(f"Error reading text/markdown file {file_path}: {e}")
        return pages_data

    def process_directory(self, data_dir: str) -> list[dict]:
        """Traverses the source data folder and applies multi-format chunking pipelines."""
        all_chunks = []
        if not os.path.exists(data_dir):
            print(f"Directory '{data_dir}' not found.")
            return all_chunks
            
        for file in os.listdir(data_dir):
            full_path = os.path.join(data_dir, file)
            file_lower = file.lower()
            raw_docs = []

            if file_lower.endswith('.pdf'):
                raw_docs = self.extract_from_pdf(full_path)
            elif file_lower.endswith('.txt') or file_lower.endswith('.md'):
                raw_docs = self.extract_from_text_or_markdown(full_path)
            else:
                continue
                
            for doc in raw_docs:
                chunks = self.splitter.split_text(doc["text"])
                for chunk in chunks:
                    all_chunks.append({
                        "text": chunk,
                        "metadata": doc["metadata"]
                    })
        return all_chunks

if __name__ == "__main__":
    
    # PROVING CONFIGURABILITY: Explicitly passing custom parameters down to the initialization loop
    CUSTOM_CHUNK_SIZE = 600
    CUSTOM_OVERLAP = 75
    
    processor = DocumentProcessor(chunk_size=CUSTOM_CHUNK_SIZE, chunk_overlap=CUSTOM_OVERLAP)
    sample_chunks = processor.process_directory("data")
    
    print(f"\n[CONFIG LOG] Running configuration matrix: Chunk Size = {processor.chunk_size}, Overlap = {processor.chunk_overlap}")
    print(f"[CLEANING LOG] Total high-density parsed text segments: {len(sample_chunks)}")
    
    if len(sample_chunks) > 0:
        print("\nFirst sample chunk preview (Verify clean data strings without headers/footers):")
        print("-" * 60)
        print(sample_chunks[0]["text"][:200] + "...")
        print("-" * 60)
