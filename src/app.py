import gradio as gr
import logging
import time

from model.rag import RAG, rag_response
from model.qwen_integration import QwenModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Initialize Qwen model
try:
    qwen_model = QwenModel(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    rag_model = RAG(model_interface=qwen_model, bm25_path="../data/processed_cooking_data.csv",
                    semantic_path="../data/processed_cooking_data.csv")
except Exception as e:
    logger.critical(f"Failed to initialize RAG model: {e}")
    raise e


def get_rag_response(user_input, retrieval_method):
    logger.info(f"Received user input: {user_input}")
    start_time = time.time()
    response_obj = rag_response(user_input, rag_model, retrieval_method=retrieval_method)
    end_time = time.time()
    logger.info(f"Generated response in {end_time - start_time:.2f} seconds.")
    return response_obj.get_response()


iface = gr.Interface(
    fn=get_rag_response,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your query here..."),
        gr.Radio(["none", "bm25", "semantic"], value="bm25", label="Retrieval Method")
    ],
    outputs="text",
    title="Qwen RAG Assistant",
    description="A Retrieval-Augmented Generation app powered by Qwen.\nSelect your retrieval method and ask a question.",
    examples=[
        ["How to bake cookies?", "bm25"],
        ["How to make bacon chewier?", "semantic"]
    ],
    flagging_mode="never",
    theme="default"
)

if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    iface.launch()
    logger.info("Gradio interface launched.")
