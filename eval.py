import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from models import StreamingSafetyHead
from dataset import SafetyDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def evaluate_safety_head(
    ckpt_path,
    test_dataset_dir,
    model_name="Qwen/Qwen3-8B",
    idx_layer=20,
    max_length=4096,
    batch_size=1,
    num_workers=2,
    bf16=True,
    device=None,
    ):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    build_cache_if_missing = True
    if build_cache_if_missing:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device
        )
        base_model.to(device=device, dtype=torch.bfloat16 if bf16 else torch.float16 if torch.cuda.is_available() else torch.float32)
        base_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = None
        base_model = None

    test_dataset = SafetyDataset(
        dataset_dir=test_dataset_dir,
        tokenizer=tokenizer,
        base_model=base_model,
        model_name=model_name,
        device="cpu" if not build_cache_if_missing else base_model.device,
        idx_layer=idx_layer,
        max_length=max_length,
        build_cache_if_missing=build_cache_if_missing,
        overwrite=False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    sample0 = torch.load(test_dataset.files[0], map_location="cpu")
    input_dim = sample0["embeddings"].shape[-1]
    
    safety_head = StreamingSafetyHead(
        input_dim=input_dim,
        proj_dim=1024,
        mem_dim=1024,
        num_labels=2,
        use_dt=True,
    )


    state = torch.load(ckpt_path, map_location="cpu")
    safety_head.load_state_dict(state, strict=True)
    safety_head.to(device=device, dtype=torch.bfloat16 if (bf16 and device.type == "cuda") else torch.float32)
    safety_head.eval()
    
    predictions = []
    references = []
    
    autocast_enabled = bf16 and (device.type == "cuda")
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        for batch in tqdm(test_loader):
            assert batch["labels"].size(0) == 1, "Current evaluation assumes batch_size=1 for streaming."
    
            labels = batch["labels"].to(device)            # (1, T_assistant)
            feat = batch["embeddings"].to(device)          # (seq, hidden)
    
            assistant_start = batch["assistant_start"]
            if isinstance(assistant_start, (list, tuple)):
                assistant_start = assistant_start[0]
            if isinstance(assistant_start, torch.Tensor):
                assistant_start = int(assistant_start.item())
            else:
                assistant_start = int(assistant_start)
    
            logits = safety_head(feat, assistant_start, )
    
            preds = logits.argmax(dim=-1)  # (1, T_assistant)
    
            preds_valid = preds.view(-1).tolist()
            labels_valid = labels.view(-1).tolist()
    
            predictions.append(preds_valid)
            references.append(labels_valid[-1])
       
    return predictions, references


if __name__=='__main__':
    model_name = "Qwen/Qwen3-8B"
    ckpt_path = "ckpts/Qwen-Qwen3-8B/seval.pt"
    test_dataset_dir = "data/s_eval/qwen3_8b/testset/"
    idx_layer = 21

    predictions, references = evaluate_safety_head(
        ckpt_path=ckpt_path,
        test_dataset_dir=test_dataset_dir,
        model_name=model_name,
        idx_layer=idx_layer,
        max_length=4096,
        batch_size=1,
        num_workers=2,
        bf16=True
    )

    print('ckpt_path: ', ckpt_path)
    print('-------------Response level-------- \n', classification_report(references, [pred[-2] for pred in predictions], digits=4))

    print('\n-----------Streaming level-----------\n', classification_report(references, [max(pred) for pred in predictions], digits=4))

