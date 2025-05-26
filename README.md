Improve the **precision of RAG (Retrieval-Augmented Generation)** using cross-encoders to rerank semantically retrieved documents.

> **Use case:** You’ve embedded your docs and built a vector search pipeline. But your top-k hits are still noisy. That’s because standard bi-encoder RAG often retrieves *relevant-looking garbage*. This repo injects a cross-encoder scoring step to **rerank results using deeper semantic understanding**, improving downstream LLM output.

![Using Cross Encoders to refine ranking on retrieved results](./assets/cross-encoder.png)

---

## ⚙️ What This Repo Does

- ✅ Extracts technical passages from the official Kubernetes docs  
- ✅ Builds a FAISS index with dense vector search  
- ✅ Fine-tunes a **cross-encoder** on query–passage pairs  
- ✅ Uses the cross-encoder to **rerank** RAG results before LLM inference  
- ✅ Fully containerized: each step runnable via Docker  
- ✅ Local dev support for faster iteration

---

## 🔥 Why Use a Cross-Encoder?

Most vector databases retrieve results based on **bi-encoder** similarity—fast, scalable, and imprecise.

A **cross-encoder**, in contrast, evaluates the relevance of a *(query, passage)* pair by jointly encoding both with full attention. That gives you **high-precision reranking** at the cost of throughput—perfect for reordering top-k hits before LLM use.

> Example:
> - Query: *"How do I create a Kubernetes Job?"*
> - Bi-encoder top-3: irrelevant references to CronJobs and DaemonSets
> - Cross-encoder reranking: pushes the actual Job creation doc to top-1

---
## 📚 Related Article

Want to understand the **why** behind this code?

Check out the accompanying article on Substack:

> **🧠 When and Why to Use Shared vs. Separate Encoders**  
> [https://sameerwadkar.substack.com/p/when-and-why-to-use-shared-vs-separate](https://sameerwadkar.substack.com/p/when-and-why-to-use-shared-vs-separate)

This article explains:
- Why general-purpose embeddings often fall short in technical or niche domains  
- How cross-encoders improve the precision of RAG systems through semantic reranking  
- When to apply thresholding to detect irrelevant retrievals  
- Why fine-tuning a cross-encoder is often a more efficient solution than retraining bi-encoders

This repository puts those ideas into practice using Kubernetes documentation as a real-world use case.

---

## 🧪 Prerequisites for Local Run

Create and activate a virtual environment:

```bash
python3 -m venv .rag-env
source .rag-env/bin/activate
pip install --upgrade pip
Install dependencies:

```bash
pip install "transformers[torch]" sentence-transformers faiss-cpu
```

---

## 📥 Download Kubernetes Documentation

Clone the docs repository:

```bash
export GIT_LOCAL_FOLDER=~/Documents
cd ${GIT_LOCAL_FOLDER}
git clone https://github.com/kubernetes/website.git
cd website/content/en/docs
```

Set the environment variable:

```bash
export DOCS_HOME_FOLDER=${GIT_LOCAL_FOLDER}
# Final path to docs: ${DOCS_HOME_FOLDER}/website/content/en/docs
```

---

> ⚠️ **Apple Silicon Users (M1/M2):**  
> Some steps may fail on ARM architecture due to missing binary support.  
> To run AMD64 Docker containers, install Rosetta 2:

```bash
/usr/sbin/softwareupdate --install-rosetta --agree-to-license
```

This enables support for x86/amd64 binaries within Docker and ensures compatibility with many Python and ML libraries that do not yet support ARM64 natively.

---

> ✅ The following steps **can** be run on ARM architecture without Docker:
> - Creating a JSON file of passages
> - Creating a FAISS Index

---

## 🔨 Workflow

You can run each step either using **Docker** or **locally**.

---

### 1️⃣ Create JSON File of Passages

**Using Docker:**

```bash
docker build --platform=linux/amd64 -f docker/CreatePassagesDockerfile -t createpassages .
docker run -e BASE_FOLDER=/app/data -v ${DOCS_HOME_FOLDER}:/app/data createpassages
```

**Run Locally:**

```bash
export BASE_FOLDER=${DOCS_HOME_FOLDER}
python scripts/create_k8s_packages_json.py
```

---

### 2️⃣ Create FAISS Index

**Using Docker:**

```bash
docker build --platform=linux/amd64 -f docker/CreateFAISSIndexDockerfile -t createfaissindex .
docker run -e BASE_FOLDER=/app/data -v ${DOCS_HOME_FOLDER}:/app/data createfaissindex
```

**Run Locally:**

```bash
export BASE_FOLDER=${DOCS_HOME_FOLDER}
python scripts/faiss_index.py
```

---

### 3️⃣ Fine-tune Cross Encoder

**Using Docker:**

```bash
docker build --platform=linux/amd64 -f docker/FineTuneCrossEncoderDockerfile -t cross-encoder-tuning-runner .
docker run -e BASE_FOLDER=/app/data -v ${DOCS_HOME_FOLDER}:/app/data cross-encoder-tuning-runner
```

**Run Locally:**

This may lead to segmentation faults on Mac
```bash

export BASE_FOLDER=${DOCS_HOME_FOLDER}
python scripts/fine_tune_cross_encoder.py
```

---

### 4️⃣ Run RAG with Cross Encoder and LLM

**Using Docker:**

```bash
docker build --platform=linux/amd64 -f docker/RAGCrossEncodeLLMDockerfile -t rag-runner .
docker run -e BASE_FOLDER=/app/data -v ${DOCS_HOME_FOLDER}:/app/data rag-runner
```


**Run Locally:**

This may lead to segmentation faults on Mac

```bash

export BASE_FOLDER=${DOCS_HOME_FOLDER}
python scripts/query_faiss_index.py
```

---

## ⚙️ Tech Note: No LangChain Used

This pipeline is implemented **without LangChain**.

All stages—embedding, FAISS indexing, cross-encoder reranking, and inference—are built directly using:

- `sentence-transformers` for embedding and cross-encoding  
- `faiss` for efficient similarity search  
- Standard Python for JSON/IO and control flow

### Why?

To maintain:
- ✅ Full transparency  
- ✅ Tight control  
- ✅ Zero framework lock-in

This approach avoids unnecessary abstraction and makes it easier to **debug**, **extend**, or **integrate** into custom pipelines.

> 🧠 **LangChain is powerful** when chaining tools or agents—  
> but here, it’s **intentionally omitted** to keep the system **clean and reproducible**.
>
> The same examples using LangChain are documented in another [README_LANGCHAIN.md](./README_LANGCHAIN.md)
