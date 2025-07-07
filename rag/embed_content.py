import faiss
import numpy as np
import pickle

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

faiss.write_index(index, "index.faiss")
with open("chunk_metadata.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Embeddings and FAISS index saved.")