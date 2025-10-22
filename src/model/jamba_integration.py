import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class JambaModel:
    """
    Integration for AI21 Jamba Reasoning 3B using Hugging Face Transformers.
    Mirrors the QwenModel interface so it can be plugged into the existing RAG pipeline.
    """

    def __init__(self, model_name: str = "ai21labs/AI21-Jamba-Reasoning-3B"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Jamba model '{self.model_name}' on device '{self.device}'")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            # Ensure pad token exists to avoid generate errors
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with safe defaults; avoid hard dependency on flash-attn
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        logger.info(f"Model '{self.model_name}' loaded successfully.")

    def generate(self, messages, max_new_tokens=4096, temperature=0.7, top_p=0.9):
        # Build chat prompt using the tokenizer's chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize inputs
        model_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(self.device)

        # Generate
        generated_ids = self.model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Remove the prompt portion and decode
        prompt_length = model_inputs["input_ids"].shape[1]
        response_ids = generated_ids[0][prompt_length:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return response_text
