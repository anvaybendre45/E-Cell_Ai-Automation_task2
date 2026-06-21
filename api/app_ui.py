import streamlit as st
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

st.set_page_config(page_title="RAG Documentation Chatbot", page_icon="🤖")
st.title("🤖 Operational Docs Assistant")
st.caption("Grounded interactive pipeline querying operational segments entirely in the cloud.")

# 1. Initialize Gemini Client
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing GEMINI_API_KEY! Please add it to your Streamlit App Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# 2. Extract Document Content cleanly
@st.cache_resource
def load_and_process_docs():
    all_text = ""
    data_dir = "data"
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".pdf"):
                try:
                    reader = PdfReader(os.path.join(data_dir, file))
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            all_text += text + "\n"
                except Exception as e:
                    st.warning(f"Could not parse {file}: {e}")
    return all_text

with st.spinner("Indexing uploaded documentation segments in the cloud..."):
    full_document_context = load_and_process_docs()

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
        with st.spinner("Analyzing documentation and generating response..."):
            
            # Construct the grounded prompt instructions passing the full text context
            system_instruction = f"""
            You are a helpful operational assistant. Answer the user's question accurately using ONLY the provided documentation context below. 
            Scan the entire context carefully to find answers, even if different phrasing or synonyms are used.
            If the answer cannot be found in the context after careful scanning, politely state: "The provided documentation does not contain information on..."
            
            DOCUMENTATION CONTEXT:
            {full_document_context[:50000]}  # Safely passes up to ~12,000 words of data text
            """
            
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config={
                        "system_instruction": system_instruction,
                        "temperature": 0.1  # Keeps responses strictly grounded to your docs
                    }
                )
                answer = response.text
            except Exception as e:
                answer = f"Error calling Gemini API: {e}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
