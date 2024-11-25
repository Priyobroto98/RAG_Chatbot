from chain import *
from ingestion import *
import streamlit as st
import sqlite3
from datetime import datetime
import uuid
from streamlit_chat import message

target_source_chunks = 25
chunk_size = 500
chunk_overlap=50

# SQLite database setup
DB_NAME = "rag_app.db"

def get_db_connection():
    """Establish a connection to the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    """Create the application logs table in the SQLite database."""
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_query TEXT,
                    gpt_response TEXT,
                    model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    """Insert a new log entry into the database."""
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    """Retrieve chat history for a given session ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "user", "content": row["user_query"]},
            {"role": "ai", "content": row["gpt_response"]}
        ])
    conn.close()
    return messages

# Initialize database
create_application_logs()

# Streamlit session state initialization
def initialize_session_state():
    """Initialize Streamlit session state."""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = None
    if "sessions" not in st.session_state:
        st.session_state["sessions"] = {}
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None
    if "llm" not in st.session_state:
        st.session_state["llm"] = None
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = None

# Handle chat functionality
def handle_chat(llm, retriever):
    """Handle user input and generate responses in a chatbot interface."""
    # Sidebar for session management
    st.sidebar.subheader("Chat Sessions")
    if st.sidebar.button("Start New Chat"):
        session_id = str(uuid.uuid4())
        st.session_state["session_id"] = session_id
        st.session_state["chat_history"] = []
        st.session_state["sessions"][session_id] = f"Session {len(st.session_state['sessions']) + 1}"

    # Display and switch between chat sessions
    session_names = st.session_state["sessions"]
    selected_session = st.sidebar.radio(
        "Select a session:", 
        options=list(session_names.keys()), 
        format_func=lambda x: session_names[x]
    )

    # Load chat history for the selected session
    if selected_session != st.session_state.get("session_id"):
        st.session_state["session_id"] = selected_session
        st.session_state["chat_history"] = get_chat_history(selected_session)

    # Display chat messages
    st.subheader("Chat History")
    for idx, message_item in enumerate(st.session_state["chat_history"]):
        if message_item["role"] == "user":
            message(message_item["content"], is_user=True, avatar_style="person", key=f"user_msg_{idx}")
        elif message_item["role"] == "ai":
            message(message_item["content"], is_user=False, avatar_style="bot", key=f"ai_msg_{idx}")

    # User input
    user_question = st.text_input("Enter your question:", key="user_input")

    if st.button("Submit", key="submit_button"):
        if user_question.strip():
            # Generate answer (simulate with a placeholder for LLM interaction)
            last_k_chat_history = get_last_k_messages(st.session_state["chat_history"], k=5)
            answer = create_router_chain(llm, retriever, last_k_chat_history, user_question)
            # Update chat history
            st.session_state["chat_history"].extend([
                {"role": "user", "content": user_question},
                {"role": "ai", "content": answer}
            ])

            # Save interaction to the database
            insert_application_logs(
                st.session_state["session_id"], 
                user_question, 
                answer, 
                "llama3-8b"
            )

            

# Main Streamlit app
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
        
        vectorstore, retriever = process_documents(uploaded_files, target_source_chunks, chunk_size, chunk_overlap)
        st.session_state["retriever"] = retriever
        st.session_state["llm"] = llm
        # Handle chat interface
        if retriever:
            handle_chat(llm, retriever)

if __name__ == "__main__":
    main()
