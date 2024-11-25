from chain import *
from ingestion import *
import streamlit as st

target_source_chunks = 25
chunk_size = 500
chunk_overlap=50

def initialize_session_state():
    """Initialize Streamlit session state."""
    # Chat history to store messages
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Initialize retriever if not already present
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None

    # Initialize llm if not already present
    if "llm" not in st.session_state:
        st.session_state["llm"] = None

    # Add other variables as needed
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = None

def main():
    st.title("Document Q&A Chatbot")
    st.sidebar.header("Upload and Process Documents")
    
    # Initialize session state
    initialize_session_state()

    # File uploader in the sidebar
    uploaded_files = st.sidebar.file_uploader(
        "Upload your documents (PDF or TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        # Process uploaded files and get retriever
        vectorstore, retriever = process_documents(uploaded_files, target_source_chunks, chunk_size, chunk_overlap)
        
        # Save retriever to session state
        st.session_state["retriever"] = retriever
        
        # Handle the chat interface
        if retriever:
            handle_chat(llm, retriever)


if __name__ == "__main__":
    main()
