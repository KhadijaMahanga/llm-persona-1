# 🤖 🤰 Mshauri wa Mama Mjamzito na Mtoto (Miezi 3 ya Kwanza ya Ujauzito)

This is a Retrieval-Augmented Generation (RAG) chatbot that answers Swahili questions about the first trimester of pregnancy using a local vector store and OpenAI GPT-3.

![Python Sript](./llm_pipeline.png)


[![Demo](07.07.2025_09.13.21_REC.gif "Streamlit Chat App - Chat Interface of our lLM")](.)

## 💡 Features

- Swahili support using multilingual embeddings
- Local FAISS vector store for fast retrieval
- Streamlit interface for chatting
- TODO: Expand to other trimesters & more topics

## 🚀 To Run

```bash
- pip install -r requirements.txt
- create .env file use .env.template 
- python rag/embed_content.py      # One-time setup
- python rag/retriever.py
- python -m rag/rag_pipeline #here you test your rag pipeline

- streamlit run app.py #this is to run a chat user interface
```
