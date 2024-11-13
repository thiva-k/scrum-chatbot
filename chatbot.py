import os
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
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

chat_session = model.start_chat()

# Initialize ChromaDB client and get or create the collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="scrum_embeddings",
    metadata={"hnsw:space": "cosine"}  # Explicitly specify distance metric
)

# Initialize the embeddings model
embeddings_model = SentenceTransformer('all-mpnet-base-v2')

def run_chatbot(user_input, messages):
    # Format the conversation history for the prompt
    conversation_history = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages[-10:]]  # Limit to last 10 messages

    try:
        # Convert embeddings to float64 explicitly
        question_embedding = np.array(embeddings_model.encode(user_input), dtype=np.float64).tolist()

        # Perform similarity search in ChromaDB collection
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3,  # Fetch top 3 matches
            include=["documents"]  # Explicitly specify what to include in results
        )

        # Safely access results
        context_items = results.get('documents', [[]])[0]
        context = "\n".join(context_items) if context_items else ""

        # Construct prompt with conversation history and context
        full_prompt = f"""Assume you are a scrum software process assisting chatbot.
        Answer only queries related to it in a professional and detailed manner:

        Context from uploaded documents, use this only as an additional input to your existing knowledge, if it is related to the query or else ignore it and use your own knowledge. Prioritize your own knowledge in any case and ignore the context from uploaded documents, if your own knowledge itself has a better answer. If the query is not related to scrum, say I cannot answer out of context or something similar:\n{context}\n\n""" + "\n".join(conversation_history)

        # Generate response
        response = chat_session.send_message(full_prompt)
        return response.text

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")  # Detailed error logging
        return f"Chatbot: An error occurred: {str(e)}"
