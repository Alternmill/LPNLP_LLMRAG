# rag.py

import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from logging.handlers import RotatingFileHandler

# Configure logging with RotatingFileHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

# Create handlers
file_handler = RotatingFileHandler("rag_app.log", maxBytes=5 * 1024 * 1024, backupCount=5)
console_handler = logging.StreamHandler()

# Create formatters and add to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class RagResponse:
    def __init__(self, sentence: str, response: str = "Default response"):
        self.sentence = sentence
        self._response = response  # Internal variable to avoid conflict with method name

    def get_response(self):
        return self._response

    def __str__(self):
        return self._response


class RAG:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing RAG with model '{self.model_name}' on device '{self.device}'.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Ensure the tokenizer has a pad_token. If not, set it to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.debug("pad_token was not set. Using eos_token as pad_token.")
        else:
            logger.debug(f"pad_token set to '{self.tokenizer.pad_token}'.")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model '{self.model_name}': {e}")
            raise e

    def generate_response(self, user_input: str, max_new_tokens: int = 300, temperature: float = 0.7,
                          top_p: float = 0.9):
        logger.info("Generating response...")
        logger.debug(f"User input: {user_input}")

        # Define system message to set the assistant's personality
        system_message = {
            "role": "system",
            "content": "You are Qwen. You are a friendly and concise assistant."
        }

        # Define user message
        user_message = {
            "role": "user",
            "content": user_input
        }

        # Combine messages
        messages = [system_message, user_message]

        # Apply chat template if available
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.debug("Applied chat template using tokenizer's apply_chat_template method.")
        except AttributeError:
            # Fallback if apply_chat_template is not available
            logger.warning("Tokenizer does not have apply_chat_template. Using manual prompt construction.")
            text = ""
            for message in messages:
                if message["role"] == "system":
                    text += f"{message['content']}\n"
                elif message["role"] == "user":
                    text += f"User: {message['content']}\nAI:"

        logger.debug(f"Final prompt for generation:\n{text}")

        # Tokenize input
        try:
            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024  # Adjust based on model's max input length
            )
            logger.debug(f"Tokenized input. input_ids shape: {model_inputs['input_ids'].shape}")
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            raise e

        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)

        # Log device information
        logger.debug(f"input_ids device: {input_ids.device}")
        logger.debug(f"attention_mask device: {attention_mask.device}")

        # Generate response
        start_time = time.time()
        try:
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            end_time = time.time()
            logger.info(f"Response generated in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise e

        # Correct Extraction: Slice based on token count
        try:
            # Calculate the number of tokens in the input prompt
            prompt_length = input_ids.shape[1]
            logger.debug(f"Number of tokens in prompt: {prompt_length}")

            # Extract only the generated tokens (excluding the prompt)
            response_ids = generated_ids[0][prompt_length:]
            logger.debug(f"Generated tokens shape: {response_ids.shape}")

            # Decode the generated tokens to get the response
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            logger.debug(f"Decoded response: {response}")
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise e

        return response


def rag_response(sentence: str, rag: RAG):
    logger.info(f"Received sentence for RAG: {sentence}")
    try:
        response = rag.generate_response(sentence)
        logger.info("Response generated successfully.")
        return RagResponse(sentence, response)
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        return RagResponse(sentence, "I'm sorry, I encountered an error while generating a response.")
