import pandas as pd
import nltk
from rank_bm25 import BM25Okapi

nltk.download('punkt')
nltk.download('punkt_tab')

class BM25Retriever:
    def __init__(self, data_path="processed_cooking_data.csv"):
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)

        # Combine question + answers + link into a single searchable text field
        # This is one way to do it; you can adjust based on your needs
        documents = []
        for i, row in self.df.iterrows():
            # Document content: question + answers + link
            # Just concatenating them into one text field
            doc_text = f"Question: {row.get('question', '')}\nAnswers: {row.get('answers', '')}\nLink: {row.get('link', '')}"
            documents.append(doc_text)

        # Tokenize documents
        self.tokenized_docs = [nltk.word_tokenize(doc.lower()) for doc in documents]

        # Initialize BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query: str, top_k=5):
        tokenized_query = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices[:top_k]:
            row = self.df.iloc[idx]
            results.append({
                "question": row.get("question", ""),
                "answers": row.get("answers", ""),
                "link": row.get("link", ""),
                "score": scores[idx]
            })
        return results


if __name__ == "__main__":
    # Test the retriever
    retriever = BM25Retriever(data_path="../../data/processed_cooking_data.csv")
    query = "How can I make my bacon chewier?"
    results = retriever.retrieve(query, top_k=3)
    for r in results:
        print("Score:", r["score"])
        print("Question:", r["question"])
        print("Answers:", r["answers"])
        print("Link:", r["link"])
        print("------")
