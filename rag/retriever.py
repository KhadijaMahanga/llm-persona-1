import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# === Load model ===
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

index = faiss.read_index("rag/index.faiss")

with open("rag/chunk_metadata.pkl", "rb") as f:
    chunk_metadata = pickle.load(f)

# === Retrieval function ===
def retrieve_chunks(query, k=10):
    """Returns top-k chunks with metadata for a given Swahili query."""
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding), k)
    
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunk_metadata):
            results.append(chunk_metadata[idx])
    return results


# === Quick test ===
if __name__ == "__main__":
    user_query = input("Swali lako kwa Kiswahili: ")
    top_chunks = retrieve_chunks(user_query)

    print("\n✅ Best chunks with content that match your query:\n")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"{i}. Chanzo: {chunk.get('source')}")
        print(chunk['text'])
        print("------\n")