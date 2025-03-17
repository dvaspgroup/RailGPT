import streamlit as st  # Must be the first import

# Set page config as the first Streamlit command
st.set_page_config(page_title="RaiLChatbot", layout="wide")

# Now you can proceed with the rest of your code
import google.generativeai as genai
import fitz  # PyMuPDF
import pdfplumber
import faiss
import os
import pickle
import datetime
import io
import logging
import requests
import json
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from google.cloud import vision, storage
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import credentials, firestore, auth
from pdf2image import convert_from_path

# Check if FIREBASE_CREDENTIALS exists
if "FIREBASE_CREDENTIALS" not in st.secrets:
    st.error("Firebase credentials not found in secrets.toml. Please add them and restart the app.")
    st.stop()

# Parse the JSON string from secrets.toml
try:
    firebase_creds = json.loads(st.secrets["FIREBASE_CREDENTIALS"])
except Exception as e:
    st.error("Failed to parse Firebase credentials. Please check your secrets.toml file.")
    st.stop()

# Initialize Firebase
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error("Failed to initialize Firebase. Please check your Firebase credentials.")
        st.stop()

# Initialize Firebase Storage
STORAGE_BUCKET_NAME = "railchatbot-cb553.appspot.com"
try:
    storage_client = storage.Client.from_service_account_info(firebase_creds)
    bucket = storage_client.bucket(STORAGE_BUCKET_NAME)
except Exception as e:
    st.error("Failed to initialize Firebase Storage. Please check your Firebase credentials.")
    st.stop()

# Initialize Firestore
db = firestore.client()

# -------------------- Gemini AI Setup --------------------
try:
    GENAI_API_KEY = st.secrets["gemini_api_key"]
except KeyError:
    st.error("Gemini API key not found in secrets.toml. Please add it and restart the app.")
    st.stop()

GENAI_MODEL = "gemini-2.0-flash"
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel(GENAI_MODEL)

# -------------------- Streamlit UI --------------------
st.title("📜 RaiLChatBot 🤖")
st.markdown("💬 **Ask me anything about the uploaded files or websites:**")

# -------------------- Sentence Transformer for Embeddings --------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------- File Storage & FAISS Index --------------------
UPLOAD_DIR = "uploaded_files"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
    faiss_index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        pdf_metadata = pickle.load(f)
else:
    faiss_index = faiss.IndexFlatL2(384)  
    pdf_metadata = {}

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="RaiLChatbot", layout="wide")

# Theme Toggle
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # Default to dark mode

# Custom CSS for Dark and Light Mode
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {'#1e1e2f' if st.session_state.theme == "dark" else '#ffffff'};
        color: {'#ffffff' if st.session_state.theme == "dark" else '#000000'};
    }}
    .stSidebar {{
        background: {'#2a2a40' if st.session_state.theme == "dark" else '#f0f2f6'};
        color: {'#ffffff' if st.session_state.theme == "dark" else '#000000'};
    }}
    .stButton > button {{
        background: {'linear-gradient(135deg, #6a11cb, #2575fc)' if st.session_state.theme == "dark" else 'linear-gradient(135deg, #2575fc, #6a11cb)'};
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
        font-weight: bold;
    }}
    .stButton > button:hover {{
        background: {'linear-gradient(135deg, #2575fc, #6a11cb)' if st.session_state.theme == "dark" else 'linear-gradient(135deg, #6a11cb, #2575fc)'};
    }}
    .stTextInput > div > div > input {{
        background: {'#2a2a40' if st.session_state.theme == "dark" else '#ffffff'};
        color: {'#ffffff' if st.session_state.theme == "dark" else '#000000'};
        border: 1px solid {'#6a11cb' if st.session_state.theme == "dark" else '#2575fc'};
        border-radius: 8px;
        padding: 10px;
    }}
    .stTextInput > div > div > input:focus {{
        border-color: {'#2575fc' if st.session_state.theme == "dark" else '#6a11cb'};
    }}
    .chat-container {{
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
    }}
    .user-message {{
        background: {'linear-gradient(135deg, #6a11cb, #2575fc)' if st.session_state.theme == "dark" else 'linear-gradient(135deg, #2575fc, #6a11cb)'};
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 75%;
        align-self: flex-end;
    }}
    .ai-message {{
        background: {'#2a2a40' if st.session_state.theme == "dark" else '#f0f2f6'};
        color: {'#ffffff' if st.session_state.theme == "dark" else '#000000'};
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 75%;
        align-self: flex-start;
    }}
    .source-text {{
        color: {'#888' if st.session_state.theme == "dark" else '#555'};
        font-size: 12px;
    }}
    .sidebar-session {{
        background: {'#2a2a40' if st.session_state.theme == "dark" else '#f0f2f6'};
        color: {'#ffffff' if st.session_state.theme == "dark" else '#000000'};
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        cursor: pointer;
    }}
    .sidebar-session:hover {{
        background: {'#3a3a50' if st.session_state.theme == "dark" else '#e0e2e6'};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Theme Toggle Button
if st.sidebar.button(f"Switch to {'Light' if st.session_state.theme == 'dark' else 'Dark'} Mode"):
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

# -------------------- Persistent Login --------------------
if "user" not in st.session_state:
    st.session_state.user = None

# -------------------- Role-Based Access --------------------
st.sidebar.title("🔑 User Authentication")
user_role = st.sidebar.selectbox("Select Role", ["User", "Admin", "Superadmin"])
st.sidebar.markdown(f"**Current Role:** {user_role}")

if user_role in ["Admin", "Superadmin"]:
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        try:
            user = auth.get_user_by_email(email)
            st.session_state.user = user
            st.sidebar.success(f"Welcome, {user.uid}!")
        except Exception as e:
            st.sidebar.error(f"Login failed: {e}")

if st.session_state.user and st.sidebar.button("Logout"):
    st.session_state.user = None
    st.sidebar.success("Logged out successfully.")

# -------------------- Session Management --------------------
def create_session(title):
    """Create a new session and store it in Firestore."""
    if st.session_state.user:
        session_data = {
            "title": title,
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
        db.collection("sessions").document(st.session_state.user.uid).collection("user_sessions").add(session_data)

def get_session_chats(session_id):
    """Retrieve chat history for a specific session."""
    if st.session_state.user:
        chats_ref = db.collection("chats").document(st.session_state.user.uid).collection("session_chats").document(session_id).collection("messages")
        chats = chats_ref.order_by("timestamp").stream()
        return [chat.to_dict() for chat in chats]
    return []

# -------------------- PDF Processing Functions --------------------
def is_valid_pdf(file_path):
    """Check if the file is a valid PDF."""
    try:
        with fitz.open(file_path) as pdf:
            return True
    except Exception as e:
        logging.error(f"Invalid PDF: {e}")
        return False

def extract_text_with_pdfplumber(file_path):
    """Extract text using pdfplumber."""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""  # Handle None returns
        return text.strip()
    except Exception as e:
        logging.error(f"pdfplumber failed to extract text: {e}")
        return None

def extract_text_with_pymupdf(file_path):
    """Extract text using PyMuPDF."""
    try:
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text() or ""  # Handle None returns
        return text.strip()
    except Exception as e:
        logging.error(f"PyMuPDF failed to extract text: {e}")
        return None

def extract_text_with_google_vision(file_path):
    """Extract text from image-based PDF using Google Cloud Vision API."""
    try:
        # Set up Google Cloud Vision credentials
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            credentials_path = "C:\\Users\\sanja\\.streamlit\\GoogleVisionAPI(OCR)\\credentials.json"
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        credentials = service_account.Credentials.from_service_account_file(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        )
        client = vision.ImageAnnotatorClient(credentials=credentials)

        text = ""

        # Convert PDF to images
        images = convert_from_path(file_path)

        for i, image in enumerate(images):
            # Save the image temporarily
            image_path = f"temp_page_{i + 1}.jpg"
            image.save(image_path, "JPEG")

            # Read the image
            with io.open(image_path, "rb") as image_file:
                content = image_file.read()

            # Perform OCR using Google Cloud Vision
            image = vision.Image(content=content)
            response = client.text_detection(image=image)

            if response.error.message:
                raise Exception(f"Google Cloud Vision Error: {response.error.message}")

            if response.text_annotations:
                text += response.text_annotations[0].description + "\n"

            # Clean up temporary image file
            os.remove(image_path)

        return text.strip()
    except Exception as e:
        logging.error(f"Google Cloud Vision failed to extract text: {e}")
        return None

def extract_text(file_path):
    """Extract text using the appropriate method based on file type."""
    if is_valid_pdf(file_path):
        # Try pdfplumber first
        text = extract_text_with_pdfplumber(file_path)
        if not text:  # Fallback to PyMuPDF
            text = extract_text_with_pymupdf(file_path)
        if not text:  # Fallback to Google Cloud Vision (OCR)
            text = extract_text_with_google_vision(file_path)
    else:
        # Treat as an image and use OCR
        text = extract_text_with_google_vision(file_path)
    return text

# -------------------- Firebase Storage Upload --------------------
def upload_to_firebase(file_path, filename):
    """Uploads file to Firebase Storage and returns the public URL."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"🚨 File not found: {file_path}")

        blob = bucket.blob(f"documents/{filename}")
        blob.upload_from_filename(file_path)

        if not blob.exists():
            raise Exception("🚨 Upload failed! Blob does not exist in Firebase.")

        blob.make_public()
        file_url = blob.public_url
        st.sidebar.success(f"✅ File uploaded: {file_url}")
        return file_url
    except Exception as e:
        st.sidebar.error(f"❌ Upload failed: {e}")
        return None

def download_from_firebase(filename, destination_path):
    """Download a file from Firebase Storage."""
    try:
        blob = bucket.blob(f"documents/{filename}")
        blob.download_to_filename(destination_path)
        st.sidebar.success(f"✅ File downloaded from Firebase: {filename}")
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Failed to download file from Firebase: {e}")
        return False

# -------------------- Website Scraping Functions --------------------
def scrape_website(url):
    """Scrape text content from a website."""
    try:
        response = requests.get(url, verify=False)  # Disable SSL verification
        soup = BeautifulSoup(response.text, "html.parser")
        website_text = soup.get_text()
        return website_text.strip()
    except Exception as e:
        logging.error(f"Error scraping website {url}: {e}")
        return None

def save_scraped_content_to_firebase(url, content):
    """Save scraped content to Firebase Storage and Firestore."""
    try:
        # Save content to a text file
        filename = f"website_content_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Upload to Firebase Storage
        file_url = upload_to_firebase(file_path, filename)
        if file_url:
            # Save metadata in Firestore
            metadata = {
                "url": url,
                "filename": filename,
                "timestamp": firestore.SERVER_TIMESTAMP,
            }
            db.collection("scraped_content").add(metadata)
            st.sidebar.success(f"✅ Scraped content from {url} uploaded to Firebase!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to save scraped content: {e}")

# -------------------- File Upload & Processing --------------------
if user_role in ["Admin", "Superadmin"] and st.session_state.user:
    st.sidebar.subheader("📂 Upload Files or URLs")
    
    # File Uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )
    
    # URL Input for Multiple Websites
    url_input = st.sidebar.text_area("Enter URLs to scrape content (one URL per line)")

    # Process Uploaded Files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            
            # Save the file locally
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract text from the file
            text = extract_text(file_path)
            if text:
                # Generate embeddings
                text_embedding = embedding_model.encode([text])

                # Add to FAISS index
                faiss_index.add(text_embedding.reshape(1, -1))
                pdf_metadata[len(pdf_metadata)] = {"file": uploaded_file.name, "text": text}

                # Upload to Firebase Storage
                file_url = upload_to_firebase(file_path, uploaded_file.name)
                if file_url:
                    st.sidebar.success(f"✅ File uploaded to Firebase: {file_url}")
            else:
                st.sidebar.error(f"❌ Failed to extract text from {uploaded_file.name}")

        # Save the updated FAISS index and metadata
        faiss.write_index(faiss_index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(pdf_metadata, f)

    # Process URL Input for Multiple Websites
    if url_input:
        urls = url_input.strip().split("\n")
        for url in urls:
            if url.strip():
                try:
                    # Scrape website content
                    website_text = scrape_website(url.strip())
                    if website_text:
                        # Save scraped content to Firebase
                        save_scraped_content_to_firebase(url.strip(), website_text)
                    else:
                        st.sidebar.error(f"❌ Failed to scrape content from {url.strip()}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error scraping website {url.strip()}: {e}")

# -------------------- Chatbot UI --------------------
st.title("📜 RaiLChatBot 🤖")
st.markdown("💬 **Ask me anything about the uploaded files or websites:**")

# Dropdown for answer source
answer_source = st.selectbox(
    "Select Answer Source",
    ["Working Model (Uploaded PDFs)", "Gemini (Uploaded PDFs)", "Gemini AI (General Knowledge)"]
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_session" not in st.session_state:
    st.session_state.current_session = None

# Chat input
query = st.text_input("Type your question here...", key="query")

if st.button("Ask", key="ask_button"):
    if not st.session_state.current_session:
        # Create a new session with the first 20 words of the query as the title
        session_title = " ".join(query.split()[:20])
        create_session(session_title)
        st.session_state.current_session = session_title

    if answer_source == "Working Model (Uploaded PDFs)":
        if faiss_index.ntotal == 0:
            st.error("⚠️ No files uploaded. Please upload a file first.")
        else:
            query_embedding = embedding_model.encode([query])
            D, I = faiss_index.search(query_embedding, k=5)
            
            retrieved_texts = []
            source_docs = set()

            for idx in I[0]:
                if idx != -1:
                    retrieved_texts.append(pdf_metadata[idx]["text"])
                    source_docs.add(pdf_metadata[idx]["file"])

            context = "\n\n".join(retrieved_texts)
            response = model.generate_content(query + "\n\nContext:\n" + context)
            answer = response.text.strip()

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            chat_entry = {
                "time": timestamp,
                "user": query,
                "bot": answer,
                "sources": ", ".join(source_docs) if source_docs else "Unknown",
            }
            st.session_state.chat_history.append(chat_entry)

            if st.session_state.user:
                db.collection("chats").document(st.session_state.user.uid).collection("session_chats").document(st.session_state.current_session).collection("messages").add({
                    **chat_entry,
                    "timestamp": firestore.SERVER_TIMESTAMP
                })

    elif answer_source == "Gemini (Uploaded PDFs)":
        if faiss_index.ntotal == 0:
            st.error("⚠️ No files uploaded. Please upload a file first.")
        else:
            query_embedding = embedding_model.encode([query])
            D, I = faiss_index.search(query_embedding, k=5)
            
            retrieved_texts = []
            source_docs = set()

            for idx in I[0]:
                if idx != -1:
                    retrieved_texts.append(pdf_metadata[idx]["text"])
                    source_docs.add(pdf_metadata[idx]["file"])

            context = "\n\n".join(retrieved_texts)
            response = model.generate_content(query + "\n\nContext:\n" + context)
            answer = response.text.strip()

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            chat_entry = {
                "time": timestamp,
                "user": query,
                "bot": answer,
                "sources": ", ".join(source_docs) if source_docs else "Unknown",
            }
            st.session_state.chat_history.append(chat_entry)

            if st.session_state.user:
                db.collection("chats").document(st.session_state.user.uid).collection("session_chats").document(st.session_state.current_session).collection("messages").add({
                    **chat_entry,
                    "timestamp": firestore.SERVER_TIMESTAMP
                })

    elif answer_source == "Gemini AI (General Knowledge)":
        response = model.generate_content(query)
        answer = response.text.strip()

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        chat_entry = {
            "time": timestamp,
            "user": query,
            "bot": answer,
            "sources": "General Knowledge",
        }
        st.session_state.chat_history.append(chat_entry)

        if st.session_state.user:
            db.collection("chats").document(st.session_state.user.uid).collection("session_chats").document(st.session_state.current_session).collection("messages").add({
                **chat_entry,
                "timestamp": firestore.SERVER_TIMESTAMP
            })

# -------------------- Sidebar Session History --------------------
if st.session_state.user:
    st.sidebar.markdown("### 📜 Session History")
    sessions_ref = db.collection("sessions").document(st.session_state.user.uid).collection("user_sessions")
    sessions = sessions_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).stream()

    for session in sessions:
        session_data = session.to_dict()
        if st.sidebar.button(f"{session_data['title']} - {session_data['time']}"):
            st.session_state.current_session = session_data["title"]
            st.session_state.chat_history = get_session_chats(session.id)

# -------------------- Chat History --------------------
st.subheader("📜 Chat History")
for chat in st.session_state.chat_history:
    st.markdown(f"**🕒 {chat['time']} | You:** {chat['user']}")
    st.markdown(f"**🤖 AI:** {chat['bot']}")
    st.markdown(f"📄 Source: {chat['sources']}")
