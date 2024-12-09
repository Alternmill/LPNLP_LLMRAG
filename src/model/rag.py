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
        """
        model_interface: an object that has a `.generate(messages)` method returning a string.
        This could be an instance of QwenModel or any similar LLM interface.
        """
        self.model_interface = model_interface

    def fake_retrieve_context(self, user_input: str):
        if "cookie" in user_input.lower():
            return [
                "To get chewy cookies, try using more brown sugar.",
                "Chilling the dough before baking can improve texture.",
                "Melted butter can help achieve a softer cookie."
            ]
        else:
            return [
                "No specific context found. But this is a placeholder.",
                "Context retrieval not implemented yet."
            ]

    def generate_response(self, user_input: str):
        context_chunks = self.fake_retrieve_context(user_input)

        context_str = ""
        for i, chunk in enumerate(context_chunks, start=1):
            context_str += f"Source {i}: {chunk}\n"

        system_message = {"role": "system", "content": "You are Qwen, a helpful assistant."}
        user_message = {"role": "user", "content": user_input}

        full_prompt = [
            {
                "role": "system",
                "content": f"You have the following context:\n{context_str}\nUse this context to help answer the user's question."
            },
            user_message
        ]

        # Now call the model_interface.generate with this constructed message list
        response = self.model_interface.generate(full_prompt)
        return response


def rag_response(sentence: str, rag: RAG):
    logger.info(f"RAG response requested for: {sentence}")
    try:
        response_str = rag.generate_response(sentence)
        return RagResponse(sentence, response_str)
    except Exception as e:
        logger.error(f"Failed to generate RAG response: {e}")
        return RagResponse(sentence, "I'm sorry, I encountered an error.")
