#example.py

import os

#Run this outside with export TRANSFORMERS_NO_TF=1
#print(os.environ["TRANSFORMERS_NO_TF"])
#os.environ["TRANSFORMERS_NO_TF"] = "1"  # Disable TensorFlow in transformers

from sentence_transformers import SentenceTransformer, CrossEncoder, util
from scipy.special import expit
# Bi-encoder (shared encoder)
bi_encoder = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v3")

query = "How to reset password"

# Assume these are passages returned by the vector database which were 
# already ingested using the bi encoding

passages = [
    "Click on 'Forgot Password' on the login screen",
    "Use a strong password with numbers and symbols"
]

# Encode separately - You don't actually do the passage_embs but this is 
# underlying reasoning behind how the actual passages were returned
# by the vector database using cosine similarity

query_emb = bi_encoder.encode(query, convert_to_tensor=True)
passage_embs = bi_encoder.encode(passages, convert_to_tensor=True)
bi_scores = util.cos_sim(query_emb, passage_embs)[0]


# Cross-encoder (joint encoding)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Encode jointly - This is something you do to decide which one of the
# returned passages are more relevant. Or to decide if they are even 
# relevant to a reasonable degree
cross_scores = cross_encoder.predict([(query, p) for p in passages])

# Convert logit to expit [0,1] interval
expit_scores = [expit(logit) for logit in cross_scores]

for i, p in enumerate(passages):
    print(f"Passage {i+1}: {p}")
    print(f"  Bi-encoder score    : {bi_scores[i].item():.4f}")
    print(f"  Cross-encoder logit score : {cross_scores[i]:.4f}")
    print(f"  Cross-encoder expit score : {expit_scores[i]:.4f}")

# Now you decide what to do next- 
# 1. If the passages are relevant beyond a certain degree let the LLM generate 
#    the output 
# 2. If the passages are not relevant past a threshold, revert to text search 
# 3. Tell the user that documents were found but they were not truly relevant and
#    and the Vector DB should be updated with more documents.
