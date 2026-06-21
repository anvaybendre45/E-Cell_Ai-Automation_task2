import streamlit as st
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

st.set_page_config(page_title="RAG Documentation Chatbot", page_icon="🤖")
st.title("🤖 Operational Docs Assistant")
st.caption("Grounded interactive pipeline querying operational segments entirely in the cloud.")

# 1. Initialize Gemini Client using cloud secrets
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing GEMINI_API_KEY! Please add it to your Streamlit App Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# 2. Build a simple, lightweight cloud-native text processor
@st.cache_resource
def load_and_process_docs():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []
    
    # Check if data folder exists relative to repo root
    data_dir = "data"
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".pdf"):
                try:
                    reader = PdfReader(os.path.join(data_dir, file))
                    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    chunks = text_splitter.split_text(text)
                    all_chunks.extend(chunks)
                except Exception as e:
                    st.warning(f"Could not parse {file}: {e}")
    return all_chunks

with st.spinner("Indexing uploaded documentation segments in the cloud..."):
    document_chunks = load_and_process_docs()

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
        with st.spinner("Analyzing segments and generating response..."):
            # Simple context matching: scan chunks for matching keywords
            relevant_context = ""
            keywords = prompt.lower().split()
            matches = [chunk for chunk in document_chunks if any(kw in chunk.lower() for kw in keywords)]
            
            # Take the top matching segments up to 4000 characters
            relevant_context = "\n---\n".join(matches[:4]) if matches else "\n".join(document_chunks[:3])

            # Construct the grounded prompt instructions
            system_instruction = f"""
            You are a helpful assistant. Answer the user's question accurately using ONLY the provided documentation context below. 
            If the answer cannot be found in the context, politely state that the documentation does not contain that information.
            
            DOCUMENTATION CONTEXT:
            {relevant_context}
            """
            
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config={"system_instruction": system_instruction}
                )
                answer = response.text
            except Exception as e:
                answer = f"Error calling Gemini API: {e}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
