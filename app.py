import streamlit as st
import os
import datetime
import io
import logging
import pickle
import requests
from bs4 import BeautifulSoup
import firebase_admin
from firebase_admin import credentials, firestore, storage, auth
import faiss
from sentence_transformers import SentenceTransformer
from pdf2image import convert_from_path
from google.cloud import vision
import pdfplumber
import fitz
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="PDF Search Engine", layout="wide")

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

# Firebase initialization
@st.cache_resource
def initialize_firebase():
    try:
        if not firebase_admin._apps:
            firebase_creds = {
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
            }
            cred = credentials.Certificate(firebase_creds)
            firebase_admin.initialize_app(cred, {
                'storageBucket': st.secrets["firebase"]["storage_bucket"]
            })
        return firestore.client(), storage.bucket()
    except Exception as e:
        logger.error(f"Firebase initialization error: {e}")
        st.error("Failed to initialize Firebase. Please check your credentials.")
        return None, None

# Initialize ML components
@st.cache_resource
def initialize_ml_components():
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        dimension = 384
        index = faiss.IndexFlatL2(dimension)
        return embedding_model, index
    except Exception as e:
        logger.error(f"ML components initialization error: {e}")
        st.error("Failed to initialize ML components.")
        return None, None

# Initialize components
db, bucket = initialize_firebase()
embedding_model, faiss_index = initialize_ml_components()

def authenticate_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.authenticated = True
        st.session_state.user_info = {'email': email, 'uid': user.uid}
        return True
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        st.error("Authentication failed. Please check your credentials.")
        return False

def extract_text_from_pdf(pdf_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name

        text_content = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)

        os.unlink(tmp_path)  # Clean up temporary file
        return '\n'.join(text_content)
    except Exception as e:
        logger.error(f"PDF text extraction error: {e}")
        st.error("Failed to extract text from PDF.")
        return None

def process_pdf(uploaded_file, user_id):
    try:
        if uploaded_file is None:
            return

        # Extract text
        text_content = extract_text_from_pdf(uploaded_file)
        if not text_content:
            st.error("No text could be extracted from the PDF.")
            return

        # Generate embedding
        embedding = embedding_model.encode([text_content])[0]
        
        # Store in Firebase
        timestamp = datetime.datetime.now().isoformat()
        file_path = f"pdfs/{user_id}/{timestamp}_{uploaded_file.name}"
        
        blob = bucket.blob(file_path)
        blob.upload_from_string(
            uploaded_file.getvalue(),
            content_type='application/pdf'
        )

        # Store metadata in Firestore
        doc_ref = db.collection('pdfs').document()
        doc_ref.set({
            'user_id': user_id,
            'filename': uploaded_file.name,
            'timestamp': timestamp,
            'storage_path': file_path,
            'embedding': embedding.tolist()
        })

        st.success("PDF processed and stored successfully!")
        
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        st.error("Failed to process PDF.")

def search_documents(query, user_id):
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode([query])[0]
        
        # Get user's documents from Firestore
        docs = db.collection('pdfs').where('user_id', '==', user_id).stream()
        
        # Collect embeddings and metadata
        embeddings = []
        metadata = []
        
        for doc in docs:
            doc_data = doc.to_dict()
            embeddings.append(doc_data['embedding'])
            metadata.append({
                'filename': doc_data['filename'],
                'storage_path': doc_data['storage_path'],
                'timestamp': doc_data['timestamp']
            })
        
        if not embeddings:
            st.info("No documents found in your library.")
            return []
        
        # Create temporary FAISS index
        temp_index = faiss.IndexFlatL2(384)
        temp_index.add(np.array(embeddings))
        
        # Search
        k = min(5, len(embeddings))  # Get top 5 results or less
        D, I = temp_index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for i in range(len(I[0])):
            if D[0][i] < 100:  # Distance threshold
                results.append(metadata[I[0][i]])
        
        return results
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        st.error("Search failed. Please try again.")
        return []

def main():
    st.title("PDF Search Engine")

    # Sidebar for authentication
    with st.sidebar:
        st.header("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_user(email, password):
                st.success("Logged in successfully!")
            else:
                st.error("Login failed")

    # Main content
    if st.session_state.authenticated:
        st.write(f"Welcome, {st.session_state.user_info['email']}")
        
        # File upload
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file is not None:
            if st.button("Process PDF"):
                process_pdf(uploaded_file, st.session_state.user_info['uid'])

        # Search interface
        st.header("Search Your Documents")
        search_query = st.text_input("Enter your search query")
        if search_query:
            if st.button("Search"):
                results = search_documents(search_query, st.session_state.user_info['uid'])
                if results:
                    for result in results:
                        st.write(f"📄 {result['filename']}")
                        st.write(f"📅 Uploaded on: {result['timestamp']}")
                        if st.button(f"Download {result['filename']}", key=result['storage_path']):
                            blob = bucket.blob(result['storage_path'])
                            url = blob.generate_signed_url(datetime.timedelta(seconds=300))
                            st.markdown(f"[Download PDF]({url})")
                else:
                    st.info("No matching documents found.")

    else:
        st.info("Please login to use the application.")

if __name__ == "__main__":
    main()
