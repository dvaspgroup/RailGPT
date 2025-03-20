import streamlit as st
import datetime

def create_session(db):
    """Create a new chat session"""
    try:
        session_ref = db.collection('chat_sessions').add({
            'user_id': st.session_state.user['localId'],
            'start_time': datetime.datetime.now(),
            'status': 'active'
        })
        return session_ref[1].id
    except Exception as e:
        st.error(f"Error creating session: {e}")
        return None

def get_session_chats(db, session_id):
    """Retrieve chats for a specific session"""
    try:
        chats = db.collection('chat_sessions').document(session_id).collection('chats').order_by('timestamp').stream()
        return [chat.to_dict() for chat in chats]
    except Exception as e:
        st.error(f"Error retrieving session chats: {e}")
        return []

def handle_session_history(db):
    """Handle chat session history"""
    try:
        st.sidebar.subheader("💭 Chat Sessions")
        
        # Create new session button
        if st.sidebar.button("New Chat Session"):
            st.session_state.current_session = create_session(db)
            st.session_state.chat_history = []
            st.rerun()

        # Get user's sessions
        sessions = db.collection('chat_sessions').where('user_id', '==', st.session_state.user['localId']).order_by('start_time', direction='DESCENDING').stream()
        
        # Display sessions
        for session in sessions:
            session_data = session.to_dict()
            session_time = session_data['start_time'].strftime("%Y-%m-%d %H:%M:%S")
            
            if st.sidebar.button(f"Session: {session_time}", key=session.id):
                st.session_state.current_session = session.id
                # Load session chats
                session_chats = get_session_chats(db, session.id)
                st.session_state.chat_history = [
                    {
                        'time': chat['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                        'user': chat['user_message'],
                        'bot': chat['ai_response'],
                        'sources': chat['sources']
                    }
                    for chat in session_chats
                ]
                st.rerun()

    except Exception as e:
        st.error(f"Error handling session history: {e}")
