# app.py

import gradio as gr
from rag import RAG, rag_response
import logging
import time

# Get the logger from rag.py
logger = logging.getLogger(__name__)

# Initialize the RAG model once
try:
    rag_model = RAG(model_name="Qwen/Qwen2.5-1.5B-Instruct")  # Replace with your specific model if different
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

# Define Gradio interface
iface = gr.Interface(
    fn=get_rag_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your message here..."),
    outputs="text",
    title="Qwen RAG Assistant",
    description="A Retrieval-Augmented Generation app powered by Qwen, your friendly and concise assistant with a touch of humor.",
    examples=[
        ["Tell me a joke about programmers."],
        ["Explain the concept of RAG in simple terms."],
        ["What's the weather like today?"]
    ],
    flagging_mode="never",
    theme="default"
)

if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    iface.launch()
    logger.info("Gradio interface launched.")
