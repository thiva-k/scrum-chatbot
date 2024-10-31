
import os
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Configure Google Generative AI API
genai.configure(api_key="AIzaSyC7Aew8RBOsdhJIVz8OD8UUtKmBfdJbayI")

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

chat_session = model.start_chat()

# Initialize ChromaDB client and get or create the collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="scrum_embeddings")

# Initialize the embeddings model
embeddings_model = SentenceTransformer('all-mpnet-base-v2')

conversation_history = []

def run_chatbot(user_input):
    conversation_history.append("User: " + user_input)

    try:
        question_embedding = embeddings_model.encode(user_input).tolist()

        # Perform similarity search in ChromaDB collection
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3  # Fetch top 3 matches
        )
        
        # Fixed the way we access results
        context_items = results['documents'][0] if results['documents'] else []
        context = "\n".join(context_items)

        # Construct prompt with conversation history and context
        full_prompt = f"""Assume you are a scrum software process assisting chatbot.
        Answer only queries related to it in a professional and detailed manner:

        Context from uploaded documents, use this only as an additional input to your existing knowledge, if it is related to the query or else ignore it and use your own knowledge. Prioritize your own knowledge in any case and ignore the context from uploaded documents, if your own knowledge itself has a better answer. If the query is not related to scrum, say I cannot answer out of context or something similar:\n{context}\n\n""" + "\n".join(conversation_history)

        # Generate response
        response = chat_session.send_message(full_prompt)
        conversation_history.append("Chatbot: " + response.text)
        return response.text

    except Exception as e:
        return f"Chatbot: An error occurred: {e}"