import streamlit as st
import datetime
import numpy as np

def search_similar_chunks(query, faiss_index, embedding_model, k=5):
    """Search for similar text chunks using FAISS"""
    query_embedding = embedding_model.encode([query])[0]
    D, I = faiss_index.search(np.array([query_embedding]), k)
    return D[0], I[0]

def format_response(response, sources):
    """Format the chat response with sources"""
    return {
        'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'user': st.session_state.get('query', ''),
        'bot': response,
        'sources': sources
    }

def handle_chat_interaction(query, answer_source, faiss_index, model, embedding_model, db):
    """Handle chat interactions based on selected answer source"""
    try:
        if not query:
            st.warning("Please enter a question.")
            return

        response = ""
        sources = ""

        if answer_source == "Working Model (Uploaded PDFs)":
            # Search similar chunks
            distances, indices = search_similar_chunks(query, faiss_index, embedding_model)
            context = "Context from documents: " + " ".join([str(idx) for idx in indices])
            
            # Generate response using context
            response = model.generate_content(f"Based on this context: {context}\n\nQuestion: {query}").text
            sources = "PDF Documents"

        elif answer_source == "Gemini (Uploaded PDFs)":
            # Similar to above but with different prompt
            distances, indices = search_similar_chunks(query, faiss_index, embedding_model)
            context = "Context from documents: " + " ".join([str(idx) for idx in indices])
            
            response = model.generate_content(
                f"Using only the following context, answer the question. If the answer isn't in the context, say so.\n\nContext: {context}\n\nQuestion: {query}"
            ).text
            sources = "Gemini + PDF Documents"

        else:  # Gemini AI (General Knowledge)
            response = model.generate_content(query).text
            sources = "Gemini AI"

        # Update chat history
        chat_response = format_response(response, sources)
        st.session_state.chat_history.append(chat_response)

        # Save to database if user is logged in
        if st.session_state.user and st.session_state.current_session:
            db.collection('chat_sessions').document(st.session_state.current_session).collection('chats').add({
                'timestamp': datetime.datetime.now(),
                'user_message': query,
                'ai_response': response,
                'sources': sources
            })

    except Exception as e:
        st.error(f"Error processing chat: {e}")
