import faiss
import numpy as np
import pickle
import os

from sentence_transformers import SentenceTransformer

# Load chunk data
with open("data/trimester1_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

texts = [chunk["text"] for chunk in chunks]
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# convert text to embeddings vectors
embeddings = model.encode(texts)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
# Save the index and the text chunks

os.makedirs("rag", exist_ok=True)
faiss.write_index(index, "rag/index.faiss")
with open("rag/chunk_metadata.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Embeddings and FAISS index saved.")