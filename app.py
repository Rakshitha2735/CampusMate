import os
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st

# File path for FAISS index
faiss_index_path = "dataset/faiss_index_cleaned.bin"

# Check if FAISS index file exists
if not os.path.exists(faiss_index_path):
    st.error("Error: FAISS index file is missing! Please upload it.")
    st.stop()  # Stop the app if file is missing

# Load FAISS index with error handling
try:
    index = faiss.read_index(faiss_index_path)
    st.success("FAISS index loaded successfully!")
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()  # Stop app if index is corrupted

# Load SentenceTransformer model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    st.success("SentenceTransformer model loaded successfully!")
except Exception as e:
    st.error(f"Error loading SentenceTransformer model: {e}")
    st.stop()

st.title("CampusMate Chatbot")
st.write("Ask me anything about KSSEM!")

# Sample input
user_input = st.text_input("Enter your question:")

if user_input:
    # Convert user input to embedding
    user_embedding = model.encode([user_input])
    
    # Search FAISS index
    D, I = index.search(user_embedding, k=5)  # Retrieve top 5 results
    
    # Display results
    st.write("Top search results:", I)
