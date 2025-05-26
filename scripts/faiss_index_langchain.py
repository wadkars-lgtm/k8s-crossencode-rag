from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from tqdm import tqdm
import os, json
from pathlib import Path

# Load documents
BASE_FOLDER = os.environ['BASE_FOLDER']
base_dir = Path(BASE_FOLDER) / "website" / "content" / "en" / "docs"

with open(base_dir / "k8s_passages.json", "r", encoding="utf-8") as f:
    passages = json.load(f)

print("✅ Done extracting passages")
documents = [Document(page_content=passage) for passage in passages]

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Texts and embeddings
texts = [doc.page_content for doc in documents]
print("⚙️ Embedding documents...")
text_embeddings = [
    (text, embedding_model.embed_query(text))
    for text in tqdm(texts, desc="Embedding", unit="doc")
]

# Build FAISS vectorstore
vectorstore = FAISS.from_embeddings(text_embeddings, embedding=embedding_model)

# Save index
vectorstore.save_local(str(base_dir / "faiss_langchain_index"))
print("✅ Done saving vector store")
