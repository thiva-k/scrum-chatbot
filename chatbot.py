import streamlit as st
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from chromadb.config import Settings

# Configure Google Generative AI API
genai.configure(api_key=os.getenv("API_KEY"))  # Replace with your actual API key

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Initialize ChromaDB client and get or create the collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="scrum_embeddings",
    metadata={"hnsw:space": "cosine"}
)

# Initialize embeddings model
embeddings_model = SentenceTransformer('all-mpnet-base-v2')

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def run_chatbot(user_input):
    st.session_state.conversation_history.append("User: " + user_input)

    try:
        question_embedding = np.array(embeddings_model.encode(user_input), dtype=np.float64).tolist()
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3,
            include=["documents"]
        )
        
        context_items = results.get('documents', [[]])[0]
        context = "\n".join(context_items) if context_items else ""

        full_prompt = f"""Assume you are a scrum software process assisting chatbot.
        Answer only queries related to it professionally:

        Context from uploaded documents (use this only if it fits the query, otherwise rely on your knowledge):\n{context}\n\n""" + "\n".join(st.session_state.conversation_history[-10:])
        
        response = model.start_chat().send_message(full_prompt)
        st.session_state.conversation_history.append("Chatbot: " + response.text)
        return response.text

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return f"Chatbot: An error occurred: {str(e)}"
