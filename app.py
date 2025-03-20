import os
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

faiss_index_path = "dataset/faiss_index_cleaned.bin"

st.title("KSSEM College Chatbot")

# Check if FAISS index file exists
if not os.path.exists(faiss_index_path):
    st.error(f"FAISS index file is missing! Expected at: {faiss_index_path}")
    st.stop()

try:
    index = faiss.read_index(faiss_index_path)
    st.success("FAISS index loaded successfully!")
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()

# Load SentenceTransformer Model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    st.success("SentenceTransformer model loaded successfully!")
except Exception as e:
    st.error(f"Error loading SentenceTransformer model: {e}")
    st.stop()

# User Input
user_input = st.text_input("Ask a question about KSSEM:")

if user_input:
    # Convert query to embedding
    query_embedding = model.encode([user_input])
    query_embedding = query_embedding.astype("float32")
    
    # Search FAISS Index
    try:
        D, I = index.search(query_embedding, k=5)  # Get top 5 matches
        
        # Display results
        st.subheader("Top Matches:")
        for i, idx in enumerate(I[0]):
            st.write(f"{i+1}. Document ID: {idx}, Similarity Score: {D[0][i]}")
    except Exception as e:
        st.error(f"Error during FAISS search: {e}")
