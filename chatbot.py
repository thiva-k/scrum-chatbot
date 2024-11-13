import os
import streamlit as st
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Configure Google Generative AI API
genai.configure(api_key=os.getenv("API_KEY"))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize session state for chat history if it doesn't exist
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    ).start_chat()

# Initialize ChromaDB client and get or create the collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="scrum_embeddings",
    metadata={"hnsw:space": "cosine"}
)

# Initialize the embeddings model
embeddings_model = SentenceTransformer('all-mpnet-base-v2')

def run_chatbot(user_input):
    # Use session state conversation history
    st.session_state.conversation_history.append("User: " + user_input)

    try:
        # Convert embeddings to float64 explicitly
        question_embedding = np.array(embeddings_model.encode(user_input), dtype=np.float64).tolist()

        # Perform similarity search in ChromaDB collection
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3,
            include=["documents"]
        )
        
        # Safely access results
        context_items = results.get('documents', [[]])[0]
        context = "\n".join(context_items) if context_items else ""

        # Construct prompt with session-specific conversation history and context
        full_prompt = f"""Assume you are a scrum software process assisting chatbot.
        Answer only queries related to it in a professional and detailed manner:

        Context from uploaded documents, use this only as an additional input to your existing knowledge, if it is related to the query or else ignore it and use your own knowledge. Prioritize your own knowledge in any case and ignore the context from uploaded documents, if your own knowledge itself has a better answer. If the query is not related to scrum, say I cannot answer out of context or something similar:\n{context}\n\n""" + "\n".join(st.session_state.conversation_history[-10:])

        # Generate response using session-specific chat
        response = st.session_state.chat_session.send_message(full_prompt)
        st.session_state.conversation_history.append("Chatbot: " + response.text)
        return response.text

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        return f"Chatbot: An error occurred: {str(e)}"

# Streamlit UI
st.title("Scrum Assistant Chatbot")

# Chat interface
user_input = st.text_input("You:", key="user_input")
if st.button("Send"):
    response = run_chatbot(user_input)
    st.write("Assistant:", response)

# Display conversation history
st.subheader("Conversation History")
for message in st.session_state.conversation_history:
    st.write(message)