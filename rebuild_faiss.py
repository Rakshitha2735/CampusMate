import faiss
import numpy as np

d = 384  # Adjust dimension to your model
index = faiss.IndexFlatL2(d)

# Generate dummy embeddings (replace with real ones)
vectors = np.random.rand(100, d).astype("float32")
index.add(vectors)

# Save FAISS index
faiss.write_index(index, "dataset/faiss_index_cleaned.bin")
print("FAISS index rebuilt successfully!")
