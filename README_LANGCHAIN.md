Improve the **precision of RAG (Retrieval-Augmented Generation)** using cross-encoders to rerank semantically retrieved documents (Using Langchain).

For design section refer to the [parent document](./README.md). This document's purpose is to serve as a runbook for examples refactored to use Langchain.

---

## 🧪 Prerequisites for Local Run

Create and activate a virtual environment:

```bash
python3 -m venv .rag-env
source .rag-env/bin/activate
pip install --upgrade pip
Install dependencies:

```bash
pip install "transformers[torch]" sentence-transformers faiss-cpu langchain langchain-community
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
docker build --platform=linux/amd64 -f docker/CreateFAISSIndexLangChainDockerfile -t createfaissindexlc .
docker run -e BASE_FOLDER=/app/data -v ${DOCS_HOME_FOLDER}:/app/data createfaissindexlc
```

**Run Locally:**

```bash
export BASE_FOLDER=${DOCS_HOME_FOLDER}
python scripts/faiss_index_lang_chain.py
```

---

### 3️⃣ Fine-tune Cross Encoder

**Using Docker:**

```bash
docker build --platform=linux/amd64 -f docker/FineTuneCrossEncoderLangChainDockerfile -t cross-encoder-tuning-runner-lc .
docker run -e BASE_FOLDER=/app/data -v ${DOCS_HOME_FOLDER}:/app/data cross-encoder-tuning-runner-lc
```

**Run Locally:**

This may lead to segmentation faults on Mac
```bash

export BASE_FOLDER=${DOCS_HOME_FOLDER}
python scripts/fine_tune_cross_encoder_lang_chain.py
```

---

### 4️⃣ Run RAG with Cross Encoder and LLM

**Using Docker:**

```bash
docker build --platform=linux/amd64 -f docker/RAGCrossEncodeLLMLangChainDockerfile -t rag-runner-lc .
docker run -e BASE_FOLDER=/app/data -v ${DOCS_HOME_FOLDER}:/app/data rag-runner-lc
```


**Run Locally:**

This may lead to segmentation faults on Mac

```bash

export BASE_FOLDER=${DOCS_HOME_FOLDER}
python scripts/query_faiss_index_langchain.py
```

---
