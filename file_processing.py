import os
import streamlit as st
import PyPDF2
import requests
from bs4 import BeautifulSoup
import numpy as np
from datetime import datetime

def process_pdf(file_path, embedding_model):
    """Extract and process text from PDF"""
    try:
        text_chunks = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                # Split text into chunks (adjust chunk size as needed)
                chunks = [text[i:i+512] for i in range(0, len(text), 512)]
                text_chunks.extend(chunks)
        return text_chunks
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

def process_url(url):
    """Scrape and process text from URL"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        # Split into chunks
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        return chunks
    except Exception as e:
        st.error(f"Error processing URL: {e}")
        return []

def process_uploaded_files(uploaded_files, upload_dir, faiss_index, embedding_model, bucket):
    """Process uploaded PDF files"""
    for uploaded_file in uploaded_files:
        try:
            # Save file locally
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Upload to Firebase Storage
            blob = bucket.blob(f"pdfs/{uploaded_file.name}")
            blob.upload_from_filename(file_path)

            # Process text
            text_chunks = process_pdf(file_path, embedding_model)
            
            # Create embeddings and add to FAISS index
            for chunk in text_chunks:
                embedding = embedding_model.encode([chunk])[0]
                faiss_index.add(np.array([embedding]))

            st.success(f"Successfully processed {uploaded_file.name}")
            
            # Clean up local file
            os.remove(file_path)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

def process_url_input(url, upload_dir, faiss_index, embedding_model):
    """Process input URL"""
    try:
        text_chunks = process_url(url)
        for chunk in text_chunks:
            embedding = embedding_model.encode([chunk])[0]
            faiss_index.add(np.array([embedding]))
        st.success(f"Successfully processed URL: {url}")
    except Exception as e:
        st.error(f"Error processing URL: {e}")
