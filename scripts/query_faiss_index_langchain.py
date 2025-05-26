import os
import json
import torch
import re
from pathlib import Path

from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------- HELPERS --------
def strip_tags(text):
    return re.sub(r'<[^>]+>', '', text)

def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,
            min_new_tokens=64,
            num_beams=3,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------- MAIN --------
def main():
    BASE_FOLDER = os.environ['BASE_FOLDER']
    base_dir = Path(BASE_FOLDER) / "website" / "content" / "en" / "docs"

    # Load metadata
    with open(base_dir / "k8s_passage_metadata.json", "r", encoding="utf-8") as f:
        passages = json.load(f)

    # Load vectorstore (LangChain FAISS wrapper)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        str(base_dir / "faiss_langchain_index"),
        embedding_model,
        allow_dangerous_deserialization=True
    )
    # Load cross encoder
    cross_encoder = CrossEncoder(str(base_dir / "fine_tuned_cross_encoder"))

    # Load generator model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model.eval()

    # Define queries
    sample_queries = [
        "How does Kubernetes handle service discovery?",
        "What are Init Containers?"
    ]

    # Loop through queries
    for query in sample_queries:
        docs = vectorstore.similarity_search(query, k=10)
        raw_passages = [strip_tags(doc.page_content) for doc in docs]

        # Re-rank with cross encoder
        pairs = [[query, p] for p in raw_passages]
        scores = cross_encoder.predict(pairs)
        reranked = [p for _, p in sorted(zip(scores, raw_passages), reverse=True)]

        # Construct prompt
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
