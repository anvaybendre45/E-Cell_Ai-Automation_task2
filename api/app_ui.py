# app_ui.py
import streamlit as st
import requests

st.set_page_config(page_title="RAG Documentation Chatbot", page_icon="🤖")
st.title("🤖 Operational Docs Assistant")
st.caption("Grounded interactive pipeline querying local vectorized operational segments.")

# 1. Initialize structural session state memory so your chat logs stay on screen
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Render previous chat threads directly from memory array cache
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Listen for fresh prompt submissions from the user chat bar
if prompt := st.chat_input("Ask a question about the operational documentation..."):
    
    # Render user prompt locally
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 4. Route payload packet to your local live FastAPI server instance
    BACKEND_URL = "http://127.0.0.1:8000/query"
    
    with st.chat_message("assistant"):
        with st.spinner("Searching segments and analyzing answers..."):
            try:
                response = requests.post(BACKEND_URL, json={"prompt": prompt}, timeout=10)
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer field found.")
                else:
                    answer = f"Error: Backend returned code {response.status_code}"
            except requests.exceptions.ConnectionError:
                answer = "Could not connect to the backend server. Make sure api/app.py is running on port 8000!"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})