import logging
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
    def __init__(self, model_interface):
        self.model_interface = model_interface
        # self.bm25_retriever = ...
        # self.semantic_retriever = ...

    def generate_response(self, user_input: str):
        system_message = {"role": "system", "content": "You are Qwen. You are a friendly and concise assistant."}
        user_message = {"role": "user", "content": user_input}
        messages = [system_message, user_message]

        response = self.model_interface.generate(messages)
        return response


def rag_response(sentence: str, rag: RAG):
    logger.info(f"RAG response requested for: {sentence}")
    try:
        response_str = rag.generate_response(sentence)
        return RagResponse(sentence, response_str)
    except Exception as e:
        logger.error(f"Failed to generate RAG response: {e}")
        return RagResponse(sentence, "I'm sorry, I encountered an error.")
