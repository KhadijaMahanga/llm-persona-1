import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import recall_score, precision_score

# Load Swahili chunks
with open("rag/chunk_metadata.pkl", "rb") as f:
    chunks = pickle.load(f)

# Extract just the 'text' for evaluation
texts = [chunk['text'] for chunk in chunks]
# Load vector store
index = faiss.read_index("rag/index.faiss")

# Load embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Evaluation Queries and Expected Chunks
evaluation_data = [
    {"query": "Dalili za mimba katika miezi mitatu ya mwanzo", "relevant_text": "Mwanamke mjamzito hupata kichefuchefu, uchovu, na kutapika katika kipindi cha trimester ya kwanza"},
    {"query": "Mabadiliko ya homoni wakati wa ujauzito", "relevant_text": "Homoni kama HCG huongezeka sana katika trimester ya kwanza"},
    {"query": "Je, ni muhimu kufanya vipimo vya afya katika trimester ya kwanza?", "relevant_text": "Vipimo vya afya ni muhimu ili kuhakikisha maendeleo mazuri ya ujauzito"},
    {"query": "Je, ni salama kula matunda yote wakati wa ujauzito?", "relevant_text": "Matunda kama parachichi ni mazuri kwa afya ya mama mjamzito, lakini mengine kama zabibu yanapaswa kuepukwa"},
    {"query": "Je, ni salama kufanya mazoezi wakati wa ujauzito?", "relevant_text": "Mazoezi mepesi kama kutembea ni salama na yanafaa kwa afya ya mama mjamzito"},
    {"query": "Je, ni salama kutumia dawa za maumivu wakati wa ujauzito?", "relevant_text": "Dawa kama paracetamol zinaweza kutumika, lakini aspirin na ibuprofen zinapaswa kuepukwa"},
    {"query": "Je, ni muhimu kufanya vipimo vya damu katika trimester ya kwanza?", "relevant_text": "Vipimo vya damu ni muhimu ili kubaini viwango vya hemoglobini na madini mengine muhimu"},
    {"query": "Je, ni salama kula vyakula vya baharini wakati wa ujauzito?", "relevant_text": "Vyakula vya baharini kama samaki wanaweza kuwa na zebaki, hivyo inashauriwa kuepuka aina fulani kama mackerel na shark"},
]


top_k = 3
recall_hits = 0
precision_hits = 0
total_queries = len(evaluation_data)

for test in evaluation_data:
    query = test["query"]
    expected = test["relevant_text"]

    # Encode the query
    query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

    # Search in FAISS index
    D, I = index.search(np.array([query_embedding]), top_k)

    retrieved_chunks = [texts[i] for i in I[0]]

    # Compute precision and recall using cosine similarity
    expected_embedding = model.encode(expected, convert_to_tensor=True)
    retrieved_embeddings = model.encode(retrieved_chunks, convert_to_tensor=True)

    cosine_scores = util.cos_sim(expected_embedding, retrieved_embeddings)[0]
    is_relevant = [score > 0.7 for score in cosine_scores]

    recall_hits += any(is_relevant)
    precision_hits += sum(is_relevant) / top_k

recall_at_k = recall_hits / total_queries
precision_at_k = precision_hits / total_queries

# Report average
print(f"Recall@{top_k}: {recall_at_k:.2f}")
print(f"Precision@{top_k}: {precision_at_k:.2f}")