import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/trimester1_chunks.csv")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# convert text to embeddings vectors
embeddings = model.encode(df['text_chunk'].tolist())
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
# Save the index and the text chunks
faiss.write_index(index, "rag/index.faiss")
df.to_pickle("rag/chunk_texts.pkl")  # Save text for retriever