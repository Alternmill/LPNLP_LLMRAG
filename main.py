# app.py

import gradio as gr
from rag import RAG, rag_response
import logging
import time

logger = logging.getLogger(__name__)

try:
    rag_model = RAG(model_name="Qwen/Qwen2.5-1.5B-Instruct")
except Exception as e:
    logger.critical(f"Failed to initialize RAG model: {e}")
    raise e


def get_rag_response(user_input):
    logger.info(f"Received user input: {user_input}")
    start_time = time.time()
    response_obj = rag_response(user_input, rag_model)
    end_time = time.time()
    logger.info(f"Generated response in {end_time - start_time:.2f} seconds.")
    return response_obj.get_response()


iface = gr.Interface(
    fn=get_rag_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your message here..."),
    outputs="text",
    title="Qwen RAG Assistant",
    description="A Retrieval-Augmented Generation app powered by Qwen, your assistant for cooking.",
    examples=[
        ["How to bake cookies."],
        ["My chicken breasts are always dry, how to make them better?."]
    ],
    flagging_mode="never",
    theme="default"
)

if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    iface.launch()
    logger.info("Gradio interface launched.")
