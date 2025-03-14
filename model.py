# model.py

import torch
import torch.nn as nn
from transformers import (
    Gemma2ForSequenceClassification
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, AdaLoraConfig



class CustomClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.fc3 = nn.Linear(config.hidden_size, config.num_labels, bias=False)

    def forward(self, x):
        x = self.fc1(x)  # 第一层全连接
        x = self.fc2(x)  # 第二层全连接
        x = self.fc3(x)  # 第三层全连接（输出类别）
        return x

class Gemma2ForSequenceClassificationCustomize(Gemma2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.score = CustomClassificationHead(config)
    




def initialize_model(config):
    model = Gemma2ForSequenceClassificationCustomize.from_pretrained(
        config.checkpoint,
        num_labels=3,
        torch_dtype=torch.bfloat16,
    )


    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.SEQ_CLS,
        init_lora_weights=config.lora_init
    )

    model = get_peft_model(model, lora_config)
    return model
