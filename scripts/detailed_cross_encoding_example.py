import os
import torch
import numpy as np
from scipy.special import expit
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# --- 1. Dual Encoder (No threshold, unchanged) ---

class DualEncoderRetriever:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.corpus_documents = []

    def index_documents(self, documents):
        print(f"Indexing {len(documents)} documents with dual encoder...")
        self.corpus_documents = documents
        self.corpus_embeddings = self.model.encode(documents, convert_to_tensor=True, show_progress_bar=True)
        print("Document indexing complete.")

    def retrieve(self, query, top_k=100):
        if self.corpus_embeddings is None or not self.corpus_documents:
            raise ValueError("Corpus not indexed. Call index_documents() first.")

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.corpus_documents)))

        retrieved_documents = []
        for score, idx in zip(top_results[0], top_results[1]):
            retrieved_documents.append({
                'document': self.corpus_documents[idx],
                'retrieval_score': score.item(),
                'index': idx.item()
            })
        return retrieved_documents

# --- 2. Cross Encoder for Ranking (Explicit Expit Conversion) ---

class CrossEncoderRanker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)
        print(f"CrossEncoder model loaded (via sentence_transformers): {model_name}")

    def rank(self, query, documents_to_rank, ranking_score_threshold=0.5):
        if not documents_to_rank:
            print("No documents provided to rank.")
            return []

        sentence_pairs = []
        original_doc_map = {}

        for i, doc in enumerate(documents_to_rank):
            doc_text = doc.get('document')
            if not isinstance(doc_text, str) or not doc_text.strip():
                print(f"Warning: Skipping problematic document at index {i} (empty or non-string): {doc_text}")
                continue
            sentence_pairs.append((query, doc_text))
            original_doc_map[len(sentence_pairs) - 1] = doc

        if not sentence_pairs:
            print("No valid sentence pairs to rank after filtering problematic documents.")
            return []

        try:
            raw_scores_from_predict = self.model.predict(sentence_pairs)
            expit_transformed_scores = expit(raw_scores_from_predict)

            if np.any(np.isnan(expit_transformed_scores)):
                print(f"CRITICAL WARNING: NaN detected in expit-transformed scores!")

        except Exception as e:
            print(f"Error during CrossEncoder prediction or expit transformation: {e}")
            print(f"Query: {query}")
            print(f"Sample Sentence Pairs that caused error (first 3): {sentence_pairs[:3]}")
            return []

        ranked_documents = []
        for i in range(len(sentence_pairs)):
            original_doc = original_doc_map[i]
            current_score = expit_transformed_scores[i].item()

            if np.isnan(current_score):
                print(f"Skipping document due to NaN score after expit: {original_doc.get('document', 'N/A')[:100]}...")
                continue

            if current_score >= ranking_score_threshold:
                original_doc['ranking_score'] = current_score
                ranked_documents.append(original_doc)
            else:
                print(f"Skipping document due to low ranking score ({current_score:.4f} < {ranking_score_threshold:.4f}): {original_doc.get('document', 'N/A')[:50]}...")

        ranked_documents.sort(key=lambda x: x['ranking_score'], reverse=True)
        return ranked_documents

# --- 3. LLM for Response Generation (Conceptual, unchanged) ---

class LLMGenerator:
    def __init__(self, llm_model_name="conceptual-llm"):
        self.llm_model_name = llm_model_name
        print(f"Initialized conceptual LLM: {self.llm_model_name}")

    def generate_response(self, query, ranked_documents, generation_params=None):
        if not ranked_documents:
            return "I couldn't find enough relevant information to answer your query after re-ranking."

        context = "\n".join([doc['document'] for doc in ranked_documents])
        prompt = f"Based on the following information, answer the question '{query}':\n\nContext:\n{context}\n\nAnswer:"

        print(f"\n--- LLM Prompt ---")
        print(prompt)
        print(f"--- End LLM Prompt ---")

        simulated_response = (
            f"The LLM, referencing the provided context about '{query}', "
            f"has synthesized the following: [Simulated summary/answer based on the top {len(ranked_documents)} documents]."
            f"The relevant points include: {context[:200]}..."
        )
        return simulated_response


# --- Example Usage ---

if __name__ == "__main__":
    # --- IMPROVED CORPUS ---
    corpus = [
        # Canine-related
        "Domesticated canines, commonly known as dogs, are loyal companions often regarded as 'man's best friend.' They exhibit a wide range of breeds with distinct characteristics and behaviors.",
        "The fox, a member of the Canidae family, is a small to medium-sized omnivorous mammal. It is known for its cunning and adaptability across various habitats.",
        "Wolves are wild canids that live in packs and are ancestors of domestic dogs. They play a crucial role in their ecosystems.",
        "Dogs thrive on companionship and regular exercise. Providing proper training and socialization from a young age is essential for their well-being.",

        # AI and Language Processing
        "Artificial intelligence (AI) is a broad field of computer science focused on creating machines that can think and learn like humans. It encompasses various subfields such as machine learning and natural language processing.",
        "Natural Language Processing (NLP) is a subfield of AI that enables computers to understand, interpret, and generate human language. NLP is vital for applications like chatbots, translation, and sentiment analysis.",
        "Machine learning, a core component of AI, involves algorithms that learn from data to make predictions or decisions without explicit programming. Deep learning is a specialized area within machine learning.",
        "The rapid advancements in artificial intelligence are transforming numerous industries, including healthcare, finance, and transportation, by automating tasks and enabling new capabilities.",

        # Small Weasels
        "Ferrets are playful, carnivorous mammals belonging to the Mustelidae family, which also includes weasels, otters, and badgers. They are known for their long, slender bodies and curious nature.",
        "Weasels are small, slender carnivorous mammals in the genus Mustela. They are agile predators known for their quick movements and voracious appetites, primarily preying on small rodents.",
        "Stoats, also known as short-tailed weasels, are members of the Mustelidae family. They are recognizable by their brown fur and a black tip on their tail, which turns white in winter in colder climates.",

        # Deep Learning
        "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (deep neural networks) to learn complex patterns from large amounts of data. It has achieved remarkable success in areas like image recognition and natural language processing.",
        "The benefits of deep learning include its ability to automatically learn features from raw data, handle large datasets, and achieve state-of-the-art performance in complex tasks where traditional machine learning struggles.",
        "Convolutional Neural Networks (CNNs) are a type of deep learning model particularly effective for image processing, while Recurrent Neural Networks (RNNs) are often used for sequential data like text.",

        # General/Less related (to observe lower scores)
        "The cat, a domestic species of small carnivorous mammal, is often valued by humans for companionship and its ability to hunt pests.",
        "Quantum computing harnesses the principles of quantum mechanics to solve problems too complex for classical computers, offering potential breakthroughs in various scientific fields.",
        "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.",
        "The history of computers dates back centuries, with early mechanical devices paving the way for modern electronic machines.",
        "This is a very short text, designed to be less informative and potentially score lower.", # Still include to see contrast
        "Sometimes sentences are short.", # Another very short one
        "Cats are known for their agility and grace.", # Existing short one
    ]

    queries = [
        "tell me about canines",
        "AI and language processing",
        "what are small weasels",
        "benefits of deep learning",
        "unrelated query about astrophysics", # Should score low
        "How do you handle special characters and very long sentences in retrieval?" # Should score low for most
    ]

    # Initialize Dual Encoder
    retriever = DualEncoderRetriever()
    retriever.index_documents(corpus)

    # Initialize Cross Encoder
    ranker = CrossEncoderRanker()

    # Initialize LLM Generator
    llm_generator = LLMGenerator()

    print("\n--- Performing Retrieval, Ranking, and LLM Generation ---")

    for query in queries:
        print(f"\n{'='*50}\nQuery: \"{query}\"\n{'='*50}")

        print("Retrieving top 5 candidates with Dual Encoder (no retrieval threshold)...")
        candidates = retriever.retrieve(query, top_k=5)

        if not candidates:
            print("No candidates retrieved by the Dual Encoder for this query.")
            print(llm_generator.generate_response(query, []))
            continue

        print("\nRetrieved Candidates (before cross-encoding ranking):")
        for i, cand in enumerate(candidates):
            print(f"{i+1}. Doc: \"{cand['document']}\" (Retrieval Score: {cand['retrieval_score']:.4f})")

        print("\nRanking candidates with Cross Encoder (ranking_threshold=0.2, explicitly using expit)...")
        # Lowering the threshold to 0.2 here so we can see more ranked results initially,
        # adjust as needed after observing scores.
        ranked_results = ranker.rank(query, candidates, ranking_score_threshold=0.2)

        if not ranked_results:
            print("No documents passed the 0.2 ranking threshold after cross-encoding.")
            print(llm_generator.generate_response(query, []))
            continue

        print("\nRanked Results (after cross-encoding and applying ranking threshold):")
        for i, result in enumerate(ranked_results):
            print(f"{i+1}. Doc: \"{result['document']}\" (Ranking Score: {result['ranking_score']:.4f})")

        print("\nGenerating response with LLM based on ranked documents...")
        llm_response = llm_generator.generate_response(query, ranked_results)
        print(f"\nLLM Generated Response:\n{llm_response}")