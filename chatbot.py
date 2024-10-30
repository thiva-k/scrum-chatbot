# chatbot.py
import os
import uuid
import google.generativeai as genai
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Configure Google Generative AI API
genai.configure(api_key="AIzaSyC5iCxiQj-NSJAVl_OVUem5tVHtwhTHlIc")

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

# Load preprocessed embeddings from the pickle file
with open("preprocessed_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
texts = data["texts"]
embeddings = data["embeddings"]
metadatas = data["metadatas"]

# Initialize the embeddings model
embeddings_model = SentenceTransformer('all-mpnet-base-v2')

conversation_history = []

def run_chatbot(user_input):
    conversation_history.append("User: " + user_input)

    try:
        question_embedding = embeddings_model.encode(user_input)

        # Simple similarity search using cosine similarity
        similarities = [
            (i, question_embedding.dot(embeddings[i]) / (np.linalg.norm(question_embedding) * np.linalg.norm(embeddings[i])))
            for i in range(len(embeddings))
        ]
        top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
        
        # Get the context from top matches
        context_items = [texts[i[0]] for i in top_matches]
        context = "\n".join(context_items)

        full_prompt = f"""Assume you are a scrum software process assisting chatbot.
        Answer only queries related to it in a professional and detailed manner:

        Context from uploaded documents, use this only as an additional input to your existing knowledge, if it is related to the query or else ignore it and use your own knowledge. Prioritize your own knowledge in any case and ignore the context from uploaded documents, if your own knowledge itself has a better answer. If the query is not related to scrum, say I cannot answer out of context or something similar:\n{context}\n\n""" + "\n".join(conversation_history)

        response = chat_session.send_message(full_prompt)
        conversation_history.append("Chatbot: " + response.text)
        return response.text
    except Exception as e:
        return None, f"Chatbot: An error occurred: {e}"