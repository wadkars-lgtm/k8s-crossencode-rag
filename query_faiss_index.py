import os
from pathlib import Path
import faiss
import json
import torch
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def strip_tags(text):
    return re.sub(r'<[^>]+>', '', text)

def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,
            min_new_tokens=64,
            num_beams=3,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    BASE_FOLDER = os.environ['BASE_FOLDER']
    base_dir = Path(BASE_FOLDER) / "website" / "content" / "en" / "docs"

    # Load FAISS index and metadata
    index = faiss.read_index(str(base_dir / "k8s_faiss.index"))
    with open(base_dir / "k8s_passage_metadata.json", "r", encoding="utf-8") as f:
        passages = json.load(f)

    # Load models
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model.eval()

    # Load fine-tuned cross encoder
    cross_encoder = CrossEncoder(Path(base_dir) / "fine_tuned_cross_encoder")

    # Define queries
    sample_queries = [
        "How does Kubernetes handle service discovery?",
        "What are Init Containers?"
    ]

    # Encode and retrieve
    query_vecs = embed_model.encode(sample_queries, convert_to_numpy=True)
    D, I = index.search(query_vecs, k=10)  # fetch more to allow re-ranking

    # Generate answers
    for idx, hits in enumerate(I):
        query = sample_queries[idx]
        candidates = [strip_tags(passages[h]) for h in hits]
        pairs = [[query, c] for c in candidates]

        # Re-rank using cross encoder
        scores = cross_encoder.predict(pairs)
        reranked = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
        context = "\n".join(reranked[:3])

        prompt = (
            f"You are a Kubernetes expert. Use the context below to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        print(f"\nQuery FAISS + CrossEncoder + Generation: {query}")
        output = generate_answer(prompt, tokenizer, model)
        print(output)
        print("\n")

if __name__ == "__main__":
    main()
