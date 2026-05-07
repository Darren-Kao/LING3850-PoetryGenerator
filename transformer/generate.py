"""
Generate quatrains using a fine-tuned GPT-2 Medium model.
Loads a saved checkpoint and completes a quatrain from a given first line.
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load model weights and tokenizer from a local checkpoint folder
def load_model(checkpoint_dir: str, device: torch.device):
    model     = GPT2LMHeadModel.from_pretrained(checkpoint_dir).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_dir)
    return model, tokenizer


# Generate a full quatrain given the first line as a prompt
def generate_from_line(
    model,
    tokenizer,
    device,
    first_line: str,
    temperature: float = 0.7,
    top_p: float = 0.92,
    top_k: int = 30,
    repetition_penalty: float = 1.4,
) -> str:
    prompt = f"<|line1|> {first_line} <|line2|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Stop generation at end-of-quatrain token or end-of-sequence token
    eos_ids = [
        tokenizer.convert_tokens_to_ids("<|endofquatrain|>"),
        tokenizer.eos_token_id,
    ]

    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=3,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.eos_token_id,
    )

    raw = tokenizer.decode(output[0], skip_special_tokens=False)
    for tok in ["<|line1|>", "<|line2|>", "<|line3|>", "<|line4|>", "<|endofquatrain|>"]:
        raw = raw.replace(tok, "\n")
    return raw.strip()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CHECKPOINT = "./model" #ADJUST THIS BASED ON WHERE FINETUNED MODEL SAVED
    model, tokenizer = load_model(CHECKPOINT, device)

    # Sample first lines — EDIT OR ADD YOUR OWN TO GENERATE NEW QUATRAINS
    first_lines = [
        "The solemn bells did mourn the dying day",
        "Beneath the frost, the waking river sighed",
        "The war-worn king laid down his crown of ash",
        "Through autumn's gold, her silent footsteps came",
        "Thy beauty mocks the lilies of the field",
    ]

    print("=" * 55)
    for i, line in enumerate(first_lines, start=1):
        print(f"\nQuatrain {i} — {line}")
        print("─" * 55)
        print(generate_from_line(model, tokenizer, device, line))
