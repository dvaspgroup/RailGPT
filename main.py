import streamlit as st
import datetime
import os
import logging
from firebase_admin import firestore
from config import setup_firebase, setup_models, initialize_storage
from auth import check_user_role, handle_authentication
from file_processing import process_uploaded_files, process_url_input
from chat import handle_chat_interaction
from session_management import create_session, get_session_chats, handle_session_history

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize Firebase and Models
auth, db, bucket = setup_firebase()
faiss_index, embedding_model, model = setup_models()
UPLOAD_DIR = initialize_storage()

def main():
    st.title("📜 RaiLChatBot 🤖")
    
    # Authentication
    user_role = handle_authentication(auth)
    
    # File Upload Section (Admin/Superadmin only)
    if user_role in ["Admin", "Superadmin"] and st.session_state.user:
        st.sidebar.subheader("📂 Upload Files or URLs")
        uploaded_files = st.sidebar.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True
        )
        url_input = st.sidebar.text_input("Enter a URL to scrape content")
        
        # Process uploads
        if uploaded_files:
            process_uploaded_files(uploaded_files, UPLOAD_DIR, faiss_index, embedding_model, bucket)
        
        # Process URL
        if url_input:
            process_url_input(url_input, UPLOAD_DIR, faiss_index, embedding_model)
    
    # Chatbot Interface
    st.markdown("💬 **Ask me anything about the uploaded files or websites:**")
    answer_source = st.selectbox(
        "Select Answer Source",
        ["Working Model (Uploaded PDFs)", "Gemini (Uploaded PDFs)", "Gemini AI (General Knowledge)"]
    )
    
    # Initialize chat state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_session" not in st.session_state:
        st.session_state.current_session = None
    
    # Chat input and processing
    query = st.text_input("Type your question here...", key="query")
    if st.button("Ask", key="ask_button"):
        handle_chat_interaction(
            query,
            answer_source,
            faiss_index,
            model,
            embedding_model,
            db
        )
    
    # Session History
    if st.session_state.user:
        handle_session_history(db)
    
    # Display Chat History
    st.subheader("📜 Chat History")
    for chat in st.session_state.chat_history:
        st.markdown(f"**🕒 {chat['time']} | You:** {chat['user']}")
        st.markdown(f"**🤖 AI:** {chat['bot']}")
        st.markdown(f"📄 Source: {chat['sources']}")

if __name__ == "__main__":
    main()
