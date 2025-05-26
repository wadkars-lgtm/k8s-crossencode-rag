import os
import json
import re
from pathlib import Path

import faiss
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------- CONFIG --------
BASE_FOLDER = os.environ['BASE_FOLDER']
base_dir = Path(BASE_FOLDER) / "website" / "content" / "en" / "docs"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
output_cross_encoder_path = str(base_dir / "fine_tuned_cross_encoder")

# -------- HELPERS --------
def strip_tags(text):
    return re.sub(r'<[^>]+>', '', text)

# -------- LOAD PASSAGES AND INDEX --------
with open(base_dir / "k8s_passage_metadata.json", "r", encoding="utf-8") as f:
    passages = json.load(f)

embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectorstore = FAISS.load_local(
    str(base_dir / "faiss_langchain_index"),
    embedding_model,
    allow_dangerous_deserialization=True
)
# -------- DEFINE SAMPLE QUERIES --------
sample_queries = [
    "How does Kubernetes handle service discovery?",
    "What are Init Containers?",
    "How does a Kubernetes Service work?",
    "What is a ConfigMap in Kubernetes?",
    "How are secrets managed in Kubernetes?",
    "What is a DaemonSet?",
    "How does Kubernetes handle persistent storage?",
    "What is a ReplicaSet?",
    "How do liveness and readiness probes work?",
    "How does Kubernetes manage container networking?"
]

# -------- FAISS SEARCH AND TRIPLET GENERATION --------
labeled_examples = []

for query in sample_queries:
    top_docs = vectorstore.similarity_search(query, k=5)
    if not top_docs or len(top_docs) < 4:
        continue

    pos = strip_tags(top_docs[0].page_content)
    negs = [strip_tags(doc.page_content) for doc in top_docs[1:4]]

    labeled_examples.append(InputExample(texts=[query, pos], label=1.0))
    for neg in negs:
        labeled_examples.append(InputExample(texts=[query, neg], label=0.0))

# -------- FINE-TUNE CROSS ENCODER --------
cross_encoder = CrossEncoder(cross_encoder_model_name, num_labels=1)
train_dataloader = DataLoader(labeled_examples, shuffle=True, batch_size=16)

cross_encoder.fit(
    train_dataloader=train_dataloader,
    epochs=1,
    show_progress_bar=True
)

# -------- SAVE MODEL --------
cross_encoder.save(output_cross_encoder_path)
print(f"Saved fine-tuned model to: {output_cross_encoder_path}")
