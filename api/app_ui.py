import streamlit as st
import os
from pypdf import PdfReader
from google import genai

st.set_page_config(page_title="RAG Documentation Chatbot", page_icon="🤖")
st.title("🤖 Operational Docs Assistant")
st.caption("Optimized lightweight contextual scanning pipeline.")

# 1. Initialize Gemini Client
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing GEMINI_API_KEY! Please add it to your Streamlit App Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# 2. Extract Document Content into manageable paragraphs
@st.cache_resource
def load_and_chunk_docs():
    paragraphs = []
    data_dir = "data"
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".pdf"):
                try:
                    reader = PdfReader(os.path.join(data_dir, file))
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            # Split by double newlines or punctuation blocks to get paragraphs
                            lines = text.split("\n\n")
                            for line in lines:
                                clean_line = line.strip()
                                if len(clean_line) > 40:  # Skip empty noise lines
                                    paragraphs.append(clean_line)
                except Exception as e:
                    pass
    return paragraphs

with st.spinner("Processing documents into lightweight segments..."):
    document_chunks = load_and_chunk_docs()

# 3. Handle Chat Layout & State
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the operational documentation..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving segments and calling assistant..."):
            
            # Smart scoring context lookup
            query_words = set(prompt.lower().split())
            scored_chunks = []
            
            for chunk in document_chunks:
                chunk_lower = chunk.lower()
                # Count how many unique question words exist in this paragraph chunk
                score = sum(1 for word in query_words if word in chunk_lower)
                if score > 0:
                    scored_chunks.append((score, chunk))
            
            # Sort by highest match score and take the top 3 snippets
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            top_matches = [item[1] for item in scored_chunks[:3]]
            
            # Fallback if no matching keywords are found at all
            if not top_matches:
                relevant_context = "\n".join(document_chunks[:2])
            else:
                relevant_context = "\n---\n".join(top_matches)

            # Grounded System Prompt
            system_instruction = f"""
            You are a helpful operational assistant. Answer the user's question accurately using ONLY the provided documentation context snippets below.
            If the answer cannot be found in the provided snippets, state: "The documentation context does not contain clear info on this query."
             Do not use any outside knowledge.
            
            DOCUMENTATION SNIPPETS:
            {relevant_context}
            """
            
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config={
                        "system_instruction": system_instruction,
                        "temperature": 0.1
                    }
                )
                answer = response.text
            except Exception as e:
                # Give a clean countdown hint if a rate limit still triggers
                if "429" in str(e):
                    answer = "⚠️ Rate limit paused. Streamlit is waiting for the free API quota window to clear. Please try your question again in 10 seconds!"
                else:
                    answer = f"Error: {e}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
