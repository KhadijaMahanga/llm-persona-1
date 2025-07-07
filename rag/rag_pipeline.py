import os

from dotenv import load_dotenv
from openai import OpenAI
from retriever import retrieve_chunks

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def answer_with_rag(user_query, model="gpt-3.5-turbo", k=3, show_sources=False):
    # === Step 1: Retrieve top-k relevant chunks ===
    retrieved_chunks = retrieve_chunks(user_query, k=k)
    
    # === Step 2: Construct context for LLM ===
    context = "\n\n".join(f"- {chunk['text']}" for chunk in retrieved_chunks)

    prompt = f"""
Uliza swali la afya ya uzazi kuhusu ujauzito katika miezi mitatu ya kwanza.

Muktadha kutoka kwenye vyanzo vya kuaminika:
{context}

Swali:
{user_query}

Jibu kwa Kiswahili fasaha na sahihi, kulingana na muktadha uliotolewa. Lakini unaweza kuongeza maarifa mengine kutoka ujuzi wako ikiwa inahitajika
"""

    # === Step 3: Send to LLM ===
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Wewe ni mtaalamu wa afya ya uzazi unayejibu maswali kwa Kiswahili."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    # === Optional: append sources ===
    if show_sources:
        sources = "\n\n".join(
            f"🔗 {chunk.get('source')}" for chunk in retrieved_chunks
        )
        answer += f"\n\nChanzo:\n{sources}"

    return answer


# === Quick test ===
if __name__ == "__main__":
    q = input("Swali lako kwa Kiswahili: ")
    print("\n🧠 Jibu:")
    print(answer_with_rag(q, show_sources=True))