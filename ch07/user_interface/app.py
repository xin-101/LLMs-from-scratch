import os
import sys
import tiktoken
import torch
import chainlit

# --- 新增：动态添加路径 ---
# 获取当前脚本所在目录 (user_interface)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录 (ch07)，这是 previous_chapters.py 所在的目录
parent_dir = os.path.dirname(current_dir)
# 将 ch07 目录添加到 Python 模块搜索路径中
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# 现在可以从 ch07 目录直接导入 previous_chapters 模块
# ------------------------

# --- 修改：直接导入 previous_chapters 模块 ---
try:
    import previous_chapters  # 直接导入 previous_chapters.py 文件
except ImportError as e:
    print(f"Error importing previous_chapters: {e}")
    sys.exit(1)
# 使用 previous_chapters.GPTModel 等
GPTModel = previous_chapters.GPTModel
generate = previous_chapters.generate
text_to_token_ids = previous_chapters.text_to_token_ids
token_ids_to_text = previous_chapters.token_ids_to_text
# ---------------------------------------------

# --- 修改：检查并打印设备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- PyTorch version: {torch.__version__} ---")
print(f"--- CUDA available: {torch.cuda.is_available()} ---")
if torch.cuda.is_available():
    print(f"--- CUDA device name: {torch.cuda.get_device_name(0)} ---")
else:
    print("--- CUDA is NOT available. Using CPU. ---")
print(f"--- Selected device: {device} ---")
# ---------------------------------------------

def get_model_and_tokenizer():
    """
    Code to load a GPT-2 model with finetuned weights generated in chapter 7.
    This requires that you run the code in chapter 7 first, which generates the necessary gpt2-medium355M-sft.pth file.
    """

    GPT_CONFIG_355M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Shortened context length (orig: 1024)
        "emb_dim": 1024,         # Embedding dimension
        "n_heads": 16,           # Number of attention heads
        "n_layers": 24,          # Number of layers
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = os.path.join(parent_dir, "gpt2-medium355M-sft.pth") # 使用 parent_dir 构建模型路径
    if not os.path.exists(model_path):
        print(
            f"Could not find the {model_path} file. Please run the chapter 7 code "
            " (ch07.ipynb) to generate the gpt2-medium355M-sft.pt file."
        )
        sys.exit()

    checkpoint = torch.load(model_path, weights_only=True, map_location=device) # Added map_location
    model = GPTModel(GPT_CONFIG_355M)
    model.load_state_dict(checkpoint)
    model.to(device)

    return tokenizer, model, GPT_CONFIG_355M


def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()


# Obtain the necessary tokenizer and model files for the chainlit function below
print("--- Loading model and tokenizer... ---")
tokenizer, model, model_config = get_model_and_tokenizer()
print("--- Model loaded successfully! ---")


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """

    torch.manual_seed(123)

    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {message.content}
    """

    # 打印本次推理的设备信息
    print(f"--- Generating on device: {device} ---")
    token_ids = generate(  # function uses `with torch.no_grad()` internally already
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=35,
        context_size=model_config["context_length"],
        eos_id=50256
    )

    text = token_ids_to_text(token_ids, tokenizer)
    response = extract_response(text, prompt)

    await chainlit.Message(
        content=f"{response}",
    ).send()