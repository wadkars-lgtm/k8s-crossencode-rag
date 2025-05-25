Improve the **precision of RAG (Retrieval-Augmented Generation)** using cross-encoders to rerank semantically retrieved documents.

> **Use case:** You’ve embedded your docs and built a vector search pipeline. But your top-k hits are still noisy. That’s because standard bi-encoder RAG often retrieves *relevant-looking garbage*. This repo injects a cross-encoder scoring step to **rerank results using deeper semantic understanding**, improving downstream LLM output.

![Using Cross Encoders to refine ranking on retrieved results](./assets/cross-encoder.png)

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

## Demonstrate Cross Encoders


```bash
export TRANSFORMERS_NO_TF=1
python basic_cross_encoding_example.py

Passage 1: Click on 'Forgot Password' on the login screen
  Bi-encoder score    : 0.6340
  Cross-encoder logit score : -2.2566
  Cross-encoder expit score : 0.0948
Passage 2: Use a strong password with numbers and symbols
  Bi-encoder score    : 0.3121
  Cross-encoder logit score : -7.4544
  Cross-encoder expit score : 0.0006
  
# To reiterate - now you decide what to do next- 
# 1. If the passages are relevant beyond a certain degree let the LLM generate 
#    the output 
# 2. If the passages are not relevant past a threshold, revert to text search 
# 3. Tell the user that documents were found but they were not truly relevant and
#    and the Vector DB should be updated with more documents.
```

Set the environment variable:

```bash
export DOCS_HOME_FOLDER=${GIT_LOCAL_FOLDER}
# Final path to docs: ${DOCS_HOME_FOLDER}/website/content/en/docs
```

Passage 1 is 150 times more relevant than Passage 2 (0.0948/0.0006) but still has a low expit score (0.0948)

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
docker build --platform=linux/amd64 -f CreatePassagesDockerfile -t createpassages .
docker run -e BASE_FOLDER=/app/data -v ${DOCS_HOME_FOLDER}:/app/data createpassages
```

**Run Locally:**

```bash
export BASE_FOLDER=${DOCS_HOME_FOLDER}
python create_k8s_packages_json.py
```

---

### 2️⃣ Create FAISS Index

**Using Docker:**

```bash
docker build --platform=linux/amd64 -f CreateFAISSIndexDockerfile -t createfaissindex .
docker run -e BASE_FOLDER=/app/data -v ${DOCS_HOME_FOLDER}:/app/data createfaissindex
```

**Run Locally:**

```bash
export BASE_FOLDER=${DOCS_HOME_FOLDER}
python faiss_index.py
```

---

### 3️⃣ Fine-tune Cross Encoder

**Using Docker:**

```bash
docker build --platform=linux/amd64 -f FineTuneCrossEncoderDockerfile -t cross-encoder-tuning-runner .
docker run -e BASE_FOLDER=/app/data -v ${DOCS_HOME_FOLDER}:/app/data cross-encoder-tuning-runner
```

**Run Locally:**

This may lead to segmentation faults on Mac
```bash

export BASE_FOLDER=${DOCS_HOME_FOLDER}
python fine_tune_cross_encoder.py
```

---

### 4️⃣ Run RAG with Cross Encoder and LLM

**Using Docker:**

```bash
docker build --platform=linux/amd64 -f RAGCrossEncodeLLMDockerfile -t rag-runner .
docker run -e BASE_FOLDER=/app/data -v ${DOCS_HOME_FOLDER}:/app/data rag-runner
```


**Run Locally:**

This may lead to segmentation faults on Mac

```bash

export BASE_FOLDER=${DOCS_HOME_FOLDER}
python query_faiss_index.py
```

---
