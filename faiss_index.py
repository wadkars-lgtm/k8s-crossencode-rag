import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Load the passages
BASE_FOLDER = os.environ['BASE_FOLDER']
# Define base directory where markdown files are located
base_dir = Path(f"{BASE_FOLDER}/website/content/en/docs")

with open(f"{base_dir}/k8s_passages.json", "r", encoding="utf-8") as f:
    passages = json.load(f)

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for the passages
embeddings = model.encode(passages, convert_to_numpy=True, show_progress_bar=True)

# Create a FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, f"{base_dir}/k8s_faiss.index")
with open(f"{base_dir}/k8s_passage_metadata.json", "w", encoding="utf-8") as f:
    json.dump(passages, f)