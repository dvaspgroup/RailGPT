import streamlit as st
from firebase_admin import auth
import datetime

def check_user_role(user_uid, db):
    """Check user role from Firestore database"""
    try:
        user_doc = db.collection('users').document(user_uid).get()
        if user_doc.exists:
            return user_doc.to_dict().get('role', 'User')
        return 'User'
    except Exception as e:
        st.error(f"Error checking user role: {e}")
        return 'User'

def handle_authentication(auth_instance):
    """Handle user authentication logic"""
    if 'user' not in st.session_state:
        st.session_state.user = None

    # Sidebar login
    with st.sidebar:
        if not st.session_state.user:
            st.subheader("👤 Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                try:
                    user = auth_instance.sign_in_with_email_and_password(email, password)
                    st.session_state.user = user
                    st.success("Login successful!")
                    st.rerun()
                except Exception as e:
                    st.error("Login failed. Please check your credentials.")
        else:
            st.write(f"👋 Welcome {st.session_state.user['email']}")
            if st.button("Logout"):
                st.session_state.user = None
                st.session_state.chat_history = []
                st.session_state.current_session = None
                st.rerun()

    # Return user role if logged in
    if st.session_state.user:
        return check_user_role(st.session_state.user['localId'], db)
    return None
