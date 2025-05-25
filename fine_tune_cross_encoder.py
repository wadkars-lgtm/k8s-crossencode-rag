from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from torch.utils.data import DataLoader
import faiss
import json
import os
from pathlib import Path
import re

# -------- CONFIG --------
BASE_FOLDER = os.environ['BASE_FOLDER']
base_dir = Path(BASE_FOLDER) / "website" / "content" / "en" / "docs"
embedding_model_name = "all-MiniLM-L6-v2"
cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
output_cross_encoder_path = f"{base_dir}/fine_tuned_cross_encoder"

# -------- HELPERS --------
def strip_tags(text):
    return re.sub(r'<[^>]+>', '', text)

# -------- LOAD INDEX AND PASSAGES --------
index = faiss.read_index(str(base_dir / "k8s_faiss.index"))
with open(base_dir / "k8s_passage_metadata.json", "r", encoding="utf-8") as f:
    passages = json.load(f)

embed_model = SentenceTransformer(embedding_model_name)

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

# -------- GENERATE EMBEDDINGS AND FAISS SEARCH --------
query_vecs = embed_model.encode(sample_queries, convert_to_numpy=True)
D, I = index.search(query_vecs, k=5)

# -------- CREATE TRIPLETS FOR CROSS ENCODER --------
labeled_examples = []

for idx, hits in enumerate(I):
    query = sample_queries[idx]
    pos = strip_tags(passages[hits[0]])
    negatives = [strip_tags(passages[h]) for h in hits[1:4]]

    labeled_examples.append(InputExample(texts=[query, pos], label=1.0))
    for neg in negatives:
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

output_cross_encoder_path