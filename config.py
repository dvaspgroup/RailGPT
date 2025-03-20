import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

def setup_firebase():
    """Initialize Firebase services"""
    cred = credentials.Certificate({
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"],
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
    })
    
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'storageBucket': st.secrets["firebase"]["storage_bucket"]
        })
    
    return auth, firestore.client(), storage.bucket()

def setup_models():
    """Initialize ML models"""
    # Initialize FAISS index
    embedding_dim = 384
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    
    # Initialize Sentence Transformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize Gemini
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-pro')
    
    return faiss_index, embedding_model, model

def initialize_storage():
    """Initialize storage directories"""
    UPLOAD_DIR = "uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    return UPLOAD_DIR
