import streamlit as st
import httpx
import json

# =====================================================================
# Configuration & Setup
# =====================================================================

st.set_page_config(page_title="RAG Project UI", page_icon="🤖", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ RAG Configuration")
    
    api_url = st.text_input("FastAPI Base URL", value="http://localhost:8000/api/v1")
    api_key = st.text_input("X-API-Key", type="password", value="dev-key", help="The API key required to access the backend.")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat history cleared!")

headers = {"X-API-Key": api_key}

# =====================================================================
# Main Application
# =====================================================================
st.title("🤖 RAG Interaction Portal")

# Tabs for different functions
tab_chat, tab_ingest = st.tabs(["💬 Chat / Query", "📂 Ingestion"])

# ---------------------------------------------------------------------
# Tab 1: Chat / Query
# ---------------------------------------------------------------------
with tab_chat:
    st.markdown("Ask questions to the RAG backend based on the ingested documents.")
    
    # Display chat messages from history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("📚 Source Chunks"):
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(f"**Source {i+1} (Score: {src['score']:.2f})**")
                        st.markdown(f"> {src['content']}")
                        st.markdown(f"*Metadata: {src['metadata']}*")

    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call backend API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Using the /query endpoint (single-turn)
                    payload = {"question": prompt, "top_k": 3}
                    r = httpx.post(f"{api_url}/query", json=payload, headers=headers, timeout=60.0)
                    r.raise_for_status()
                    data = r.json()
                    
                    answer = data.get("answer", "No answer provided.")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("📚 Source Chunks"):
                            for i, src in enumerate(sources):
                                st.markdown(f"**Source {i+1} (Score: {src['score']:.2f})**")
                                st.markdown(f"> {src['content']}")
                                st.markdown(f"*Metadata: {src['metadata']}*")
                    
                    # Log to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 403:
                        st.error("Error 403: Invalid API Key. Please update it in the sidebar.")
                    else:
                        st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

# ---------------------------------------------------------------------
# Tab 2: Ingestion
# ---------------------------------------------------------------------
with tab_ingest:
    st.header("Upload New Knowledge")
    st.markdown("Add new PDF files, text files, or Web URLs to the vector database.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. File Upload")
        uploaded_file = st.file_uploader("Upload a PDF or TXT file")
        if st.button("Upload & Ingest", type="primary"):
            if not uploaded_file:
                st.warning("Please select a file first.")
            else:
                with st.spinner("Uploading to Server..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        r = httpx.post(f"{api_url}/ingest/upload", files=files, headers=headers, timeout=30.0)
                        r.raise_for_status()
                        st.success(f"Success: {r.json().get('message')}")
                        st.info("The file is being processed in the background.")
                    except httpx.HTTPError as e:
                        st.error(f"Failed to upload: {e}")

    with col2:
        st.subheader("2. URL or Local Path")
        path_input = st.text_input("Enter Web URL (https://...) or absolute local file path")
        if st.button("Ingest Path/URL", type="secondary"):
            if not path_input.strip():
                st.warning("Please enter a valid path or URL.")
            else:
                with st.spinner("Sending request..."):
                    try:
                        payload = {"source_path": path_input.strip()}
                        r = httpx.post(f"{api_url}/ingest/file", json=payload, headers=headers, timeout=10.0)
                        r.raise_for_status()
                        st.success(f"Success: {r.json().get('message')}")
                        st.info("The URL is being processed in the background.")
                    except httpx.HTTPError as e:
                        st.error(f"Failed: {e}")
