import gradio as gr
import logging
import time

from model.rag import RAG, rag_response
from model.qwen_integration import QwenModel
from model.jamba_integration import JambaModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Initialize model registry and corresponding RAG instances
try:
    AVAILABLE_MODELS = {
        "Qwen/Qwen2.5-1.5B-Instruct": QwenModel(model_name="Qwen/Qwen2.5-1.5B-Instruct"),
        "ai21labs/AI21-Jamba-Reasoning-3B": JambaModel(model_name="ai21labs/AI21-Jamba-Reasoning-3B"),
    }
    RAG_BY_MODEL = {
        name: RAG(model_interface=mdl, bm25_path="../data/processed_cooking_data.csv", semantic_path="../data/processed_cooking_data.csv")
        for name, mdl in AVAILABLE_MODELS.items()
    }
    DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
except Exception as e:
    logger.critical(f"Failed to initialize models or RAG instances: {e}")
    raise e


def get_rag_response(user_input, retrieval_method, model_name):
    logger.info(f"Received user input: {user_input}")
    start_time = time.time()
    selected_model = model_name if model_name in RAG_BY_MODEL else DEFAULT_MODEL_NAME
    if selected_model != model_name:
        logger.warning(f"Unknown model '{model_name}', falling back to default '{DEFAULT_MODEL_NAME}'.")
    rag_instance = RAG_BY_MODEL[selected_model]
    response_obj = rag_response(user_input, rag_instance, retrieval_method=retrieval_method)
    end_time = time.time()
    logger.info(f"Generated response in {end_time - start_time:.2f} seconds using model '{selected_model}'.")
    return response_obj.get_response()


import os
import csv
from datetime import datetime

# Feedback handling
FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "../data/feedback.csv")

def _ensure_feedback_header(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not os.path.exists(path):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "vote", "query", "response", "model", "retrieval"])  # header


def submit_feedback(state, vote):
    try:
        _ensure_feedback_header(FEEDBACK_FILE)
        data = state or {}
        with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                vote,
                data.get("query", ""),
                data.get("response", ""),
                data.get("model", ""),
                data.get("retrieval", "")
            ])
        return "Thanks for the feedback!"
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        return "Failed to save feedback."


def on_submit(query, retrieval_method, model_name, state):
    resp = get_rag_response(query, retrieval_method, model_name)
    # Preserve any existing state keys and update the tracked fields
    new_state = dict(state or {})
    new_state.update({
        "query": query,
        "response": resp,
        "model": model_name,
        "retrieval": retrieval_method,
    })
    return resp, new_state


def begin_submit(query, retrieval_method, model_name, state):
    # Set a visible status and store start timestamp in state
    start_ts = time.time()
    st = dict(state or {})
    st["__start_ts__"] = start_ts
    human_ts = datetime.now().strftime("%H:%M:%S")
    return f"⏳ Working… started at {human_ts}", st


def end_submit(state):
    st = state or {}
    start_ts = st.get("__start_ts__")
    elapsed = 0.0
    if isinstance(start_ts, (int, float)):
        elapsed = max(0.0, time.time() - start_ts)
    return f"✅ Done in {elapsed:.2f} seconds"


import json
import hashlib
import binascii

# Simple local user management (JSON file with salted+hashed passwords)
USERS_FILE = os.path.join(os.path.dirname(__file__), "../data/users.json")


def _ensure_users_file(path: str = USERS_FILE):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"users": []}, f)


def _load_users(path: str = USERS_FILE) -> dict:
    _ensure_users_file(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load users file: {e}")
        return {"users": []}


def _save_users(data: dict, path: str = USERS_FILE):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save users file: {e}")


def _username_exists(data: dict, username: str) -> bool:
    uname = (username or "").strip()
    for u in data.get("users", []):
        if u.get("username", "").strip().lower() == uname.lower():
            return True
    return False


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return binascii.hexlify(salt).decode(), binascii.hexlify(dk).decode()


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = (username or "").strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password or "") < 6:
        return False, "Password must be at least 6 characters."

    data = _load_users()
    if _username_exists(data, username):
        return False, "Username already exists."

    salt_hex, hash_hex = _hash_password(password)
    data.setdefault("users", []).append({
        "username": username,
        "salt": salt_hex,
        "pw": hash_hex,
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    _save_users(data)
    return True, "Signup successful."


def authenticate_user(username: str, password: str) -> bool:
    username = (username or "").strip()
    data = _load_users()
    for u in data.get("users", []):
        if u.get("username", "").strip().lower() == username.lower():
            try:
                salt = binascii.unhexlify(u.get("salt", ""))
                expected = u.get("pw", "")
                _, hash_hex = _hash_password(password or "", salt)
                return hash_hex == expected
            except Exception as e:
                logger.error(f"Auth verify failed: {e}")
                return False
    return False


# UI with Login/Signup and gated main app
with gr.Blocks(title="Qwen RAG Assistant") as iface:
    gr.Markdown("""
    # Qwen RAG Assistant
    Please log in or sign up to use the assistant.
    """)

    state = gr.State({})  # holds { "auth_user": "...", plus query/response/model/retrieval }

    # Login / Signup area
    auth_group = gr.Group(visible=True)
    with auth_group:
        with gr.Tabs():
            with gr.Tab("Login"):
                login_user = gr.Textbox(label="Username")
                login_pass = gr.Textbox(label="Password", type="password")
                login_btn = gr.Button("Log in", variant="primary")
                login_status = gr.Markdown(visible=True)
            with gr.Tab("Sign up"):
                su_user = gr.Textbox(label="Username")
                su_pass = gr.Textbox(label="Password", type="password")
                su_pass2 = gr.Textbox(label="Confirm Password", type="password")
                signup_btn = gr.Button("Create account", variant="secondary")
                signup_status = gr.Markdown(visible=True)

    # Main app area (hidden until logged in)
    main_group = gr.Group(visible=False)
    with main_group:
        header_row = gr.Row()
        with header_row:
            user_info = gr.Markdown("", elem_id="user_info")
            logout_btn = gr.Button("Logout")

        gr.Markdown("Select your retrieval method and model, then ask a question.")

        with gr.Row():
            query = gr.Textbox(lines=2, placeholder="Enter your query here...", label="Your question")
        with gr.Row():
            retrieval = gr.Radio(["none", "bm25", "semantic"], value="bm25", label="Retrieval Method")
            model = gr.Dropdown(choices=list(AVAILABLE_MODELS.keys()), value=DEFAULT_MODEL_NAME, label="Model")
        with gr.Row():
            submit = gr.Button("Submit", variant="primary")
        with gr.Row():
            status_md = gr.Markdown(label="Status")
        with gr.Row():
            output = gr.Markdown(label="Answer")

        with gr.Row():
            like_btn = gr.Button("👍 Like")
            dislike_btn = gr.Button("👎 Dislike")
            feedback_status = gr.Markdown(visible=True)

        gr.Examples(
            examples=[
                ["How to bake cookies?", "bm25", DEFAULT_MODEL_NAME],
                ["How to make bacon chewier?", "semantic", DEFAULT_MODEL_NAME],
                ["What can I substitute for buttermilk in pancakes?", "bm25", DEFAULT_MODEL_NAME],
            ],
            inputs=[query, retrieval, model],
            label="Examples"
        )

        submit.click(fn=begin_submit, inputs=[query, retrieval, model, state], outputs=[status_md, state]) \
              .then(fn=on_submit, inputs=[query, retrieval, model, state], outputs=[output, state]) \
              .then(fn=end_submit, inputs=[state], outputs=[status_md])

        like_btn.click(fn=lambda s: submit_feedback(s, "like"), inputs=[state], outputs=[feedback_status])
        dislike_btn.click(fn=lambda s: submit_feedback(s, "dislike"), inputs=[state], outputs=[feedback_status])

    # Event handlers for auth
    def _do_login(username, password, st):
        if authenticate_user(username, password):
            new_st = dict(st or {})
            new_st["auth_user"] = username.strip()
            return gr.update(visible=False), gr.update(visible=True), new_st, f"✅ Logged in as {username}", f"**User:** {username}"
        else:
            return gr.update(visible=True), gr.update(visible=False), st, "❌ Invalid username or password.", gr.update(value="")

    def _do_signup(username, pw1, pw2, st):
        if (pw1 or "") != (pw2 or ""):
            return gr.update(visible=True), gr.update(visible=False), st, "❌ Passwords do not match.", gr.update(value="")
        ok, msg = register_user(username, pw1)
        if ok:
            new_st = dict(st or {})
            new_st["auth_user"] = username.strip()
            return gr.update(visible=False), gr.update(visible=True), new_st, f"✅ {msg} Logged in as {username}.", f"**User:** {username}"
        else:
            return gr.update(visible=True), gr.update(visible=False), st, f"❌ {msg}", gr.update(value="")

    def _do_logout(st):
        new_st = dict(st or {})
        if "auth_user" in new_st:
            del new_st["auth_user"]
        # Clear user indicator and show auth area
        return gr.update(visible=True), gr.update(visible=False), new_st, gr.update(value=""), gr.update(value=""), gr.update(value="")

    login_btn.click(_do_login, inputs=[login_user, login_pass, state], outputs=[auth_group, main_group, state, login_status, user_info])
    signup_btn.click(_do_signup, inputs=[su_user, su_pass, su_pass2, state], outputs=[auth_group, main_group, state, signup_status, user_info])
    logout_btn.click(_do_logout, inputs=[state], outputs=[auth_group, main_group, state, user_info, login_status, signup_status])


if __name__ == "__main__":
    logger.info("Launching Gradio Blocks interface...")
    iface.launch(share=True)
    logger.info("Gradio interface launched.")
