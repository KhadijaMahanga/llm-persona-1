import streamlit as st
from rag.rag_pipeline import answer_with_rag

st.set_page_config(page_title="Mama Care: Trimester 1", page_icon="🤰", layout="centered")

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤰 Mshauri wa Mama Mjamzito na Mtoto (Miezi 3 ya Kwanza ya Ujauzito)")

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])


# Accept user input
user_input = st.chat_input("Uliza swali lolote kuhusu ujauzito (miezi mitatu ya mwanzo) kwa Kiswahili.")
if user_input:
    # Show user's message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Inatafakari..."):
        try:
            response = answer_with_rag(user_input, k=5, show_sources=True)
        except Exception as e:
            response = "Samahani, kuna hitilafu: " + str(e)

    # Show assistant's reply
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
