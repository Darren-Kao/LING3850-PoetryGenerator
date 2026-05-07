"""
Train a GPT-2 Medium model to generate quatrains.
Fine-tunes on a tokenized quatrain dataset with early stopping and checkpointing.
"""

import os
import math
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# Free ~15% speed on A100/T4
torch.backends.cuda.matmul.allow_tf32 = True

DATA_PATH = "quatrain_train_tokenized.txt"

# Training hyperparameters
@dataclass
class Args:
    data_path:    str   = DATA_PATH
    output_dir:   str   = "./gpt2-quatrain-v2"
    model_name:   str   = "gpt2-medium"
    epochs:       int   = 20
    batch_size:   int   = 8
    max_length:   int   = 128
    lr:           float = 5e-5
    warmup_steps: int   = 100
    save_every:   int   = 3
    grad_accum:   int   = 2
    seed:         int   = 42
    patience:     int   = 5

#Dataset: loads and tokenizes quatrain examples from a txt file
class QuatrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        with open(data_path) as f:
            self.examples = [l.strip() for l in f if l.strip()]
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.examples[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask":enc["attention_mask"].squeeze(),
            "labels": input_ids.clone(),
        }


#Tokenizer setup: adds special tokens for quatrain structure
def build_tokenizer(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {
        "additional_special_tokens": [
            "<|line1|>", "<|line2|>", "<|line3|>",
            "<|line4|>", "<|endofquatrain|>",
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    print(f"Vocab size: {len(tokenizer)} (base 50257 + 5 special)")
    return tokenizer


#Main training loop
def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}\n  GPT-2 Quatrain Finetuner")
    print(f"  Device : {device} | Model: {args.model_name}")
    print(f"  Epochs : {args.epochs} | LR: {args.lr}\n{'='*55}")

    tokenizer = build_tokenizer(args.model_name)
    dataset   = QuatrainDataset(args.data_path, tokenizer, args.max_length)
    print(f"{len(dataset)} training examples")

    #shuffle training for each epoch = better generalization
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {total_params:,}")

    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss, no_improve = float("inf"), 0

    #Actual training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader, start=1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss   = epoch_loss / len(loader)
        perplexity = math.exp(avg_loss)
        print(f"  Epoch {epoch:>2}/{args.epochs} | loss: {avg_loss:.4f} | ppl: {perplexity:.2f}")

        if avg_loss < best_loss:
            best_loss, no_improve = avg_loss, 0
            best_dir = os.path.join(args.output_dir, "best")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"             → new best — saved to {best_dir}")
        else:
            no_improve += 1
            print(f"             → no improvement ({no_improve}/{args.patience})")
            if no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

        if epoch % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nDone. Final model → {final_dir}")
    return model, tokenizer


if __name__ == "__main__":
    args = Args()
    print(args)
    model, tokenizer = train(args)
