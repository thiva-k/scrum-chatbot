import os
import uuid
import PyPDF2
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Initialize SentenceTransformer embeddings model
embeddings_model = SentenceTransformer('all-mpnet-base-v2')

# List of PDF file paths
pdf_filepaths = [
    "C:/Users/ASUS/Downloads/ResearchScrum.pdf",  # Replace with your actual PDF paths
    "C:/Users/ASUS/Downloads/Scrum Whitepaper_web.pdf",  # Add more PDFs as needed
    # Add other PDF paths as necessary
]

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""  # Handle None text case
    return text

def process_and_embed_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    embeddings = [embeddings_model.encode(t) for t in texts]
    metadatas = [{"source": pdf_path, "text": t} for t in texts]
    ids = [str(uuid.uuid4()) for _ in texts]
    return {"texts": texts, "embeddings": embeddings, "metadatas": metadatas, "ids": ids}

# Initialize lists to hold combined data
combined_data = {
    "texts": [],
    "embeddings": [],
    "metadatas": [],
    "ids": []
}

# Process each PDF file
for pdf_filepath in pdf_filepaths:
    data = process_and_embed_pdf(pdf_filepath)
    combined_data["texts"].extend(data["texts"])
    combined_data["embeddings"].extend(data["embeddings"])
    combined_data["metadatas"].extend(data["metadatas"])
    combined_data["ids"].extend(data["ids"])
    print(f"Processed {pdf_filepath}.")

# Save combined embeddings to a .pkl file
with open("preprocessed_embeddings.pkl", "wb") as f:
    pickle.dump(combined_data, f)

print("Combined embeddings and text data saved to 'preprocessed_embeddings.pkl'.")
