import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig 
from sklearn.metrics import f1_score, classification_report
from models import StreamingSafetyHead
import math
from transformers import get_cosine_schedule_with_warmup


from tqdm import tqdm
import random
import numpy as np
import argparse

from dataset import SafetyDataset
from eval import evaluate_safety_head


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    print(f"Random seed set globally to {seed}")

set_seed(42)


def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params / 1_000_000


def remove_none_fields(d):
    if isinstance(d, dict):
        return {k: remove_none_fields(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [remove_none_fields(i) for i in d]
    else:
        return d

def compute_temporal_tv_monotone_loss(logits, valid_mask=None, lam_tv=0.01, lam_mono=0.01):
    # logits: (B, T, C)
    # valid_mask: (B, T) bool
    if logits.size(1) < 2:
        return torch.zeros([], device=logits.device, dtype=logits.dtype)

    p = torch.softmax(logits, dim=-1)[..., 1]  # (B, T)
    diffs = p[:, 1:] - p[:, :-1]               # (B, T-1)

    if valid_mask is not None:
        vm = valid_mask[:, 1:] & valid_mask[:, :-1]  # (B, T-1)
        diffs = diffs[vm]

    if diffs.numel() == 0:
        return torch.zeros([], device=logits.device, dtype=logits.dtype)

    tv = diffs.abs().mean()
    mono = torch.relu(-diffs).mean()

    return lam_tv * tv + lam_mono * mono



def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = True

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    train_dataset = SafetyDataset(
        dataset_dir=args.train_dataset_dir, 
        tokenizer=tokenizer,
        base_model=base_model,
        model_name=args.model_name,
        device=device,
        idx_layer=args.idx_layer,
        max_length=args.max_length,
        build_cache_if_missing=True,
        overwrite=False,
        max_build_samples=None
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    input_dim = AutoConfig.from_pretrained(args.model_name).hidden_size

    del base_model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    safety_head = StreamingSafetyHead(
            input_dim=input_dim,
            proj_dim=1024, 
            mem_dim=1024, 
            num_labels=2, 
            use_dt=True)


    safety_head.to(device=device, dtype=torch.bfloat16)
    safety_head.requires_grad = True

    print("Total trainable parameters: ", count_parameters(safety_head), 'M')

    optimizer = AdamW(
        safety_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    max_grad_norm = 1.0

    max_steps = -1
    lr_scheduler_type = "cosine"
    warmup_ratio = 0.05
    warmup_steps = 0

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_acc_steps)
    if max_steps is None or max_steps < 0:
        total_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        total_training_steps = max_steps

    if warmup_steps and warmup_steps > 0:
        computed_warmup_steps = warmup_steps
    else:
        computed_warmup_steps = int(total_training_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=computed_warmup_steps,
        num_training_steps=total_training_steps
    )

    os.makedirs(args.save_dir, exist_ok=True)


    global_step = 0
    completed_steps = 0
    safety_head.train()

    for epoch in range(args.num_train_epochs):
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")):
            assert batch["labels"].size(0) == 1, "Current implementation assumes batch_size_per_device=1 for streaming."
            labels = batch["labels"].to(device)  # (1, T_assistant)
            feat = batch['embeddings'].to(device)  # (seq, hidden) on CPU -> move to device

            assistant_start = batch['assistant_start']
            if isinstance(assistant_start, (list, tuple)):
                assistant_start = assistant_start[0]
            if isinstance(assistant_start, torch.Tensor):
                assistant_start = int(assistant_start.item())
            else:
                assistant_start = int(assistant_start)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):

                
                logits = safety_head(feat, assistant_start) # [Bs, N, D]
                
                loss_ce = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                anchor_mask = (labels != -100) # (B, T_assistant)
                reg_mask = torch.ones_like(anchor_mask, dtype=torch.bool)
                loss_smooth = compute_temporal_tv_monotone_loss(
                            logits, valid_mask=reg_mask, lam_tv=0.01, lam_mono=0.01
                            )

                loss = loss_ce + loss_smooth


                loss = loss / args.gradient_acc_steps

            loss.backward()

            with torch.no_grad():
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)  # (1, T_assistant)
                mask = (labels != -100)
                correct = (preds[mask] == labels[mask]).sum().item()
                total_correct += correct
                total_tokens += mask.sum().item()

            if (step + 1) % args.gradient_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(safety_head.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                completed_steps += 1
                global_step += 1

                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = total_loss / args.gradient_acc_steps
                avg_acc = (total_correct / total_tokens) if total_tokens > 0 else 0.0
                print(f"Epoch [{epoch+1}/{args.num_train_epochs}], "
                    f"UpdateStep [{completed_steps}/{total_training_steps}], "
                    f"LR: {current_lr:.2e}, Loss: {avg_loss:.4f}, Acc(token): {avg_acc:.4f}")

                total_loss = 0.0
                total_correct = 0
                total_tokens = 0

                if max_steps is not None and max_steps > 0 and completed_steps >= max_steps:
                    break

        if max_steps is not None and max_steps > 0 and completed_steps >= max_steps:
            print("Reached max_steps. Stopping training.")
            break

        ckpt_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pt")
        torch.save(safety_head.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    print("Training complete!")


    predictions, references = evaluate_safety_head(
        ckpt_path=ckpt_path,
        test_dataset_dir=args.test_dataset_dir,
        model_name=args.model_name,
        idx_layer=args.idx_layer,
        max_length=4096,
        batch_size=1,
        num_workers=2,
        bf16=True
    )

    print('ckpt_path: ', ckpt_path)
    print('-------------Response level-------- \n', classification_report(references, [pred[-2] for pred in predictions], digits=4))

    print('\n-----------Streaming level-----------\n', classification_report(references, [max(pred) for pred in predictions], digits=4))




def main():
    parser = argparse.ArgumentParser(description="Train the StreamingSafetyHead with your model.")

    # --- Model & Path ---
    parser.add_argument(
        "--train_dataset_dir",
        type=str,
        required=True,
        help="Path to the training dataset."
    )
    parser.add_argument(
        "--test_dataset_dir",
        type=str,
        required=True,
        help="Path to the test dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Path or Hugging Face ID of the base model."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to save trained model."
    )

    # --- Training recipe ---
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_acc_steps",
        type=int,
        default=32,
        help="batch size."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="the max sequence length."
    )
    parser.add_argument(
        "--idx_layer",
        type=int,
        default=32,
        help="Index of the transformer layers to use for feature extraction"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="weight decay"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="nums of training epochs"
    )

    args = parser.parse_args()

    train(args)
