from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ------------------------------
# Step 1: Small Dataset
# ------------------------------
documents = [
    "Artificial Intelligence is transforming industries",
    "Python is widely used for data analysis",
    "Machine learning finds patterns in data",
    "Football is a popular sport worldwide",
    "Deep learning is part of artificial intelligence"
]

# ------------------------------
# Step 2: Load Embedding Model
# ------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------
# Step 3: Generate Embeddings
# ------------------------------
doc_embeddings = model.encode(documents)

print("\nDOCUMENT EMBEDDINGS\n")

for i, emb in enumerate(doc_embeddings):
    print(f"Document {i+1}: {documents[i]}")
    print(f"Embedding length: {len(emb)}")
    print(f"First 10 values: {emb[:10]}\n")

# ------------------------------
# Step 4: User Query
# ------------------------------
query = input("\nEnter search query: ")

# ------------------------------
# Step 5: Convert Query to Embedding
# ------------------------------
query_embedding = model.encode([query])

# ------------------------------
# Step 6: Calculate Similarity
# ------------------------------
similarity_scores = cosine_similarity(query_embedding, doc_embeddings)

# ------------------------------
# Step 7: Find Best Match
# ------------------------------
best_match_index = np.argmax(similarity_scores)

print("\nRESULT\n")
print("Query:", query)
print("Best Matching Document:", documents[best_match_index])
print("Similarity Score:", similarity_scores[0][best_match_index])