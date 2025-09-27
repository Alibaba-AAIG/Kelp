import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob

from torch.utils.data import Dataset
from datasets import load_from_disk

from tqdm import tqdm

import random
import numpy as np
import tempfile

def find_sequence(lst, seq):
    n = len(seq)
    for i in range(len(lst) - n + 1):
        if lst[i:i+n] == seq:
            return i
    return -1


class SafetyDataset(Dataset):
    """
    Per-sample cached dataset:
    - cache_dir: {dataset_dir}/safety_cache/{model_name}/idx{idx_layer}maxlength{max_length}/
    - sample files: sample{orig_idx:08d}.pt, containing:
    {'embeddings': Tensor[seq, hidden], 'assistant_start': int, 'labels': Tensor[T_assistant]}
    - Build only on rank=0, others wait for barrier and just read.
    """
    def __init__(self,
        dataset_dir,
        model_name,
        tokenizer=None,
        base_model=None,
        idx_layer: int = 20,
        max_length: int = 4096,
        device: str = "cpu",
        build_cache_if_missing: bool = False,
        overwrite: bool = False,
        max_build_samples: int | None = None,
        debug_limit: int | None = None
        ):
        self.dataset_dir = dataset_dir
        self.model_name = model_name
        self.idx_layer = idx_layer
        self.max_length = max_length
        self.device = device
        
        # self.user_prompt_marker = [151645, 198, 151644, 77091, 198]
        self.assistant_tokens = '<|im_start|>assistant\n'
        self.assistant_end = -1
        self.num_supervised_token = 10
        self.cache_dir = os.path.join(
                dataset_dir,
                f"safety_cache/{model_name.replace('/', '-')}/idx{idx_layer}_maxlength{max_length}"
            )

        os.makedirs(self.cache_dir, exist_ok=True)
        need_build = (len(glob.glob(os.path.join(self.cache_dir, "sample_*.pt"))) == 0)
        if need_build and build_cache_if_missing:
            assert tokenizer is not None and base_model is not None, "Building cache requires tokenizer and base_model."
            self._build_cache_per_sample(
                tokenizer=tokenizer,
                base_model=base_model,
                overwrite=overwrite,
                max_build_samples=max_build_samples
            )
    
        self.files = sorted(glob.glob(os.path.join(self.cache_dir, "sample_*.pt")))
        if debug_limit is not None:
            self.files = self.files[:debug_limit]
    
        if len(self.files) == 0:
            raise FileNotFoundError(f"No cached samples found in {self.cache_dir}. "
                                    f"Set build_cache_if_missing=True on rank=0 to build first.")
    
    def _build_cache_per_sample(self, tokenizer, base_model, overwrite=False, max_build_samples=None):
        print(f"Building per-sample cache into {self.cache_dir} ...")
        data = load_from_disk(self.dataset_dir)
        total = len(data) if max_build_samples is None else min(len(data), max_build_samples)
    
        base_model.eval()
        with torch.no_grad():
            for i in tqdm(range(total), desc="Build samples"):
                sample_path = os.path.join(self.cache_dir, f"sample_{i:08d}.pt")
                if (not overwrite) and os.path.exists(sample_path):
                    continue
    
                info = data[i]
                messages = [{'role':'user', 'content': info['prompt']}, {'role':'assistant', 'content': info['response']}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    max_length=self.max_length,
                    truncation=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(self.device)
                label = info['label']
    
                output = base_model.generate(
                    **model_inputs,
                    max_new_tokens=1,
                    temperature=0,
                    top_p=1.0,
                    top_k=0,
                    do_sample=False,
                    repetition_penalty=1.0,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
                hidden_states = output.hidden_states[0][self.idx_layer]  # (1, seq, hidden)
    
                # user_to_assistant_pos = find_sequence(model_inputs.input_ids[0].tolist(), self.user_prompt_marker)

                # if user_to_assistant_pos < 0:
                #     continue
                # assistant_start = user_to_assistant_pos + len(self.user_prompt_marker)
                assistant_ids = tokenizer.encode(self.assistant_tokens)
                assistant_start = find_sequence(model_inputs.input_ids[0].tolist(), assistant_ids) + len(assistant_ids)
    
                seq_len = model_inputs.input_ids[:, assistant_start:self.assistant_end].shape[-1]
                if seq_len <= 0:
                    continue
    
                labels = torch.full((1, seq_len), -100, dtype=torch.long, device=self.device)
                labels[:, :self.num_supervised_token] = 0
                labels[:, -self.num_supervised_token:] = torch.tensor([label], device=self.device).unsqueeze(1).expand(-1, self.num_supervised_token)
    
                embedding_cpu = hidden_states[0, :self.assistant_end, :].detach().cpu().contiguous()
                labels_cpu = labels[0].detach().cpu().contiguous()
    
                payload = {
                    "embeddings": embedding_cpu,          # (seq, hidden)
                    "assistant_start": int(assistant_start),
                    "labels": labels_cpu                   # (T_assistant,)
                }
    
                tmp_fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir)
                os.close(tmp_fd)
                torch.save(payload, tmp_path)
                os.replace(tmp_path, sample_path)
    
        print(f"Cache build finished at {self.cache_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        obj = torch.load(self.files[idx], map_location="cpu")
        embeddings = obj["embeddings"]            # (seq, hidden), cpu tensor
        assistant_start = obj['assistant_start']
        labels = torch.as_tensor(obj["labels"], dtype=torch.long)  # (T_assistant)
        return {
            "embeddings": embeddings,
            "assistant_start": assistant_start,
            "labels": labels
        }


