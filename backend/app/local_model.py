"""
Local Fine-tuned Model - Medical Chatbot

Qwen2.5-3B-Instruct + LoRA adapter kullanarak lokal inference.
"""

import torch
from pathlib import Path
from typing import Optional

# Lazy loading - import only when needed
_model = None
_tokenizer = None
_device = None


def get_device():
    """Get available device (CUDA > CPU). MPS skipped - 3B model too large."""
    if torch.cuda.is_available():
        return "cuda"
    # MPS has 4GB tensor limit, 3B model exceeds it
    return "cpu"


def load_model(adapter_path: str = None):
    """
    Load the fine-tuned model with LoRA adapter.

    Args:
        adapter_path: Path to the LoRA adapter directory

    Returns:
        Tuple of (model, tokenizer)
    """
    global _model, _tokenizer, _device

    if _model is not None:
        return _model, _tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    _device = get_device()
    print(f"[LOCAL-MODEL] Loading on device: {_device}")

    # Default adapter path
    if adapter_path is None:
        adapter_path = Path(__file__).parent.parent.parent / "kaggle_training" / "checkpoint-1000"

    adapter_path = Path(adapter_path)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    print(f"[LOCAL-MODEL] Loading adapter from: {adapter_path}")

    base_model_name = "Qwen/Qwen2.5-3B-Instruct"

    # Load tokenizer first
    _tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Load base model
    print(f"[LOCAL-MODEL] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Move to device (only CUDA, CPU stays on CPU)
    if _device == "cuda":
        base_model = base_model.to("cuda")

    # Load LoRA adapter
    print(f"[LOCAL-MODEL] Loading LoRA adapter...")
    _model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    )
    _model.eval()

    print(f"[LOCAL-MODEL] Model loaded successfully!")
    return _model, _tokenizer


def generate_response(
    messages: list,
    system_prompt: str = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate response using the local fine-tuned model.

    Args:
        messages: List of message dicts with 'role' and 'content'
        system_prompt: Optional system prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        Generated response text
    """
    model, tokenizer = load_model()

    # Build chat messages
    chat_messages = []

    if system_prompt:
        chat_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        chat_messages.append({"role": msg["role"], "content": msg["content"]})

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()


# Quick test
if __name__ == "__main__":
    print("Testing local model...")

    response = generate_response(
        messages=[{"role": "user", "content": "What are symptoms of headache?"}],
        system_prompt="You are a helpful medical assistant."
    )

    print(f"Response: {response}")
