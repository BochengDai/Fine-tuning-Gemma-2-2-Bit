# config.py

from dataclasses import dataclass

@dataclass
class Config:
    output_dir: str = "../tf-logs/kaggle_ft/"
    checkpoint: str = './autodl-tmp/LLM-Research/gemma-2-2b-it'
    max_length: int = 3072
    n_splits: int = 5
    fold_idx: int = 0
    optim_type: str = "adamw_8bit"
    per_device_train_batch_size: int = 2

    gradient_accumulation_steps: int = 16
    per_device_eval_batch_size: int = 4
    n_epochs: int = 1
    freeze_layers: int = 0
    lr: float = 1e-4
    warmup_steps: int = 20
    lora_r: int = 64
    lora_alpha: float = 64
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_init: str = "pissa_niter_4"
    
