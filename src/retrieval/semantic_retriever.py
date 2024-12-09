import pandas as pd
import nltk
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    def __init__(self, data_path="../../data/processed_cooking_data.csv", model_name="all-MiniLM-L6-v2"):
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)
        self.model_name = model_name

        # Initialize the embedding model
        self.embedder = SentenceTransformer(self.model_name)

        # Combine question + answers + link into a single text field
        documents = []
        for i, row in self.df.iterrows():
            doc_text = f"Question: {row.get('question', '')}\nAnswers: {row.get('answers', '')}\nLink: {row.get('link', '')}"
            documents.append(doc_text)

        # Compute embeddings
        # This might take a while depending on the size of the dataset
        self.embeddings = self.embedder.encode(documents, convert_to_numpy=True)

        # Normalize embeddings for cosine similarity (recommended)
        faiss.normalize_L2(self.embeddings)

        # Create a FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Using inner product for similarity
        self.index.add(self.embeddings)

    def retrieve(self, query: str, top_k=5):
        # Embed the query
        query_embedding = self.embedder.encode(query, convert_to_numpy=True).reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search the index
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            # idx is the index of the retrieved document
            row = self.df.iloc[idx]
            results.append({
                "question": row.get("question", ""),
                "answers": row.get("answers", ""),
                "link": row.get("link", ""),
                "score": float(dist)
            })

        return results


if __name__ == "__main__":
    # Test the semantic retriever
    retriever = SemanticRetriever(data_path="../../data/processed_cooking_data.csv", model_name="all-MiniLM-L6-v2")
    query = "How can I make my bacon chewier?"
    results = retriever.retrieve(query, top_k=3)
    for r in results:
        print("Score:", r["score"])
        print("Question:", r["question"])
        print("Answers:", r["answers"])
        print("Link:", r["link"])
        print("------")
