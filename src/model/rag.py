import logging
import time

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.semantic_retriever import SemanticRetriever

logger = logging.getLogger(__name__)


class RagResponse:
    def __init__(self, sentence: str, response: str = "Default response"):
        self.sentence = sentence
        self._response = response

    def get_response(self):
        return self._response

    def __str__(self):
        return self._response


class RAG:
    def __init__(self, model_interface, bm25_path="../../data/processed_cooking_data.csv",
                 semantic_path="../../data/processed_cooking_data.csv"):
        self.model_interface = model_interface
        self.bm25_retriever = BM25Retriever(data_path=bm25_path)
        self.semantic_retriever = SemanticRetriever(data_path=semantic_path)

    def retrieve_docs(self, query, method='bm25', top_k=3):
        start_time = time.time()
        if method == 'bm25':
            docs = self.bm25_retriever.retrieve(query, top_k=top_k)
        elif method == 'semantic':
            docs = self.semantic_retriever.retrieve(query, top_k=top_k)
        else:
            docs = []  # no retrieval

        end_time = time.time()
        logger.info(f"Retrieval method: {method}, retrieved {len(docs)} docs in {end_time - start_time:.2f} seconds.")

        # Log the retrieved context
        for i, d in enumerate(docs, start=1):
            logger.info(f"Doc {i}: Q:{d['question']} A:{d['answers']} L:{d['link']}")

        return docs

    def generate_response(self, user_input: str, retrieval_method='bm25'):
        # Retrieve documents
        docs = self.retrieve_docs(user_input, method=retrieval_method, top_k=3)

        # Build context string
        context_str = ""
        sources_str = ""
        for i, doc in enumerate(docs, start=1):
            context_str += f"Source {i}:\nQuestion: {doc['question']}\nAnswers: {doc['answers']}\nLink: {doc['link']}\n\n"
            # We'll keep track of sources for citation
            if doc['link']:
                sources_str += f"[{i}]: {doc['link']}\n"

        # Add instructions to return links. The prompt instructs the model to cite sources at the end.
        system_message = {"role": "system",
                          "content": "You are Qwen, a helpful assistant that uses the given cooking advice as context."}
        user_message = {"role": "user", "content": user_input}

        # We include the context as an additional piece of information in the system message or a separate role.
        # To ensure the model sees all tokens, we can append the context after the system and user messages:
        assistant_message = {
            "role": "assistant",
            "content": f"CONTEXT:\n{context_str}\nPlease answer the user's question using the context above. At the end of your answer, cite the sources as [1], [2], etc., and list them at the end like:\n\nSources:\n[sources here]\n"
        }

        messages = [system_message, user_message, assistant_message]

        response = self.model_interface.generate(messages)

        # Append sources after response:
        final_response = response.strip() + "\n\nSources:\n" + (
            sources_str.strip() if sources_str.strip() else "No sources found.")
        return final_response


def rag_response(sentence: str, rag: RAG, retrieval_method='bm25'):
    logger.info(f"RAG response requested for: {sentence} with method={retrieval_method}")
    try:
        response_str = rag.generate_response(sentence, retrieval_method=retrieval_method)
        return RagResponse(sentence, response_str)
    except Exception as e:
        logger.error(f"Failed to generate RAG response: {e}")
        return RagResponse(sentence, "I'm sorry, I encountered an error.")
