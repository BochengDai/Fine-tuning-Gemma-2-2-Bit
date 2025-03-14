import json
import time
from tqdm import tqdm
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import sklearn
import numpy as np
import pandas as pd
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast, BitsAndBytesConfig
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import PeftModel

# assert torch.cuda.device_count() == 2 # 用于确保你是否有2个gpu

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
        # self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

@dataclass
class Config:
    gemma_dir = '/home/PaulDai/Project_milestone/sysllm/sysllm/autodl-tmp/LLM-Research/gemma-2-2b-it'
    lora_dir = '/home/PaulDai/Project_milestone/sysllm/tf-logs/kaggle_ft_0302/checkpoint-1796/' # 替换这个ckpt为你的ckpt即可
    max_length = 2048
    batch_size = 8
    device = torch.device("cuda")    
    tta = True  # test time augmentation. <prompt>-<model-b's response>-<model-a's response>
    spread_max_length = False  # whether to apply max_length//3 on each input or max_length on the concatenated input

cfg = Config()

test = pd.read_csv('/home/PaulDai/Project_milestone/sysllm/sysllm/test.csv') # 将这个文件替换为test.csv所在路径

def process_text(text: str) -> str:
    try:
        return " ".join(json.loads(text))  # 确保解析后是字符串
    except:
        return str(text)  # 确保不会返回 None
test.loc[:, 'prompt'] = test['prompt'].apply(process_text)
test.loc[:, 'response_a'] = test['response_a'].apply(process_text)
test.loc[:, 'response_b'] = test['response_b'].apply(process_text)

# display(test.head(5))

def tokenize(
    tokenizer, prompt, response_a, response_b, max_length=cfg.max_length, spread_max_length=cfg.spread_max_length
):
    prompt = ["<prompt>: " + p for p in prompt]
    response_a = ["\n\n<response_a>: " + r_a for r_a in response_a]
    response_b = ["\n\n<response_b>: " + r_b for r_b in response_b]
    if spread_max_length:

        prompt = tokenizer(prompt, max_length=max_length//3, truncation=True, padding=False).input_ids
        response_a = tokenizer(response_a, max_length=max_length//3, truncation=True, padding=False).input_ids
        response_b = tokenizer(response_b, max_length=max_length//3, truncation=True, padding=False).input_ids
        input_ids = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        attention_mask = [[1]* len(i) for i in input_ids]
    else:
        text = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = tokenizer(text, max_length=max_length, truncation=True, padding=False)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
    return input_ids, attention_mask


tokenizer = GemmaTokenizerFast.from_pretrained(cfg.gemma_dir)
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"

data = pd.DataFrame()
data["id"] = test["id"]
data["input_ids"], data["attention_mask"] = tokenize(tokenizer, test["prompt"], test["response_a"], test["response_b"])
data["length"] = data["input_ids"].apply(len)

aug_data = pd.DataFrame()
aug_data["id"] = test["id"]
# swap response_a & response_b
aug_data['input_ids'], aug_data['attention_mask'] = tokenize(tokenizer, test["prompt"], test["response_b"], test["response_a"])
aug_data["length"] = aug_data["input_ids"].apply(len)

# 在单GPU上运行
print("loading model")
device = torch.device('cuda:0')
model = Gemma2ForSequenceClassificationCustomize.from_pretrained(
    cfg.gemma_dir,
    device_map=device,
    use_cache=False,
    num_labels=3,
)


model = PeftModel.from_pretrained(model, cfg.lora_dir)
print("model load")


@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, device, batch_size=cfg.batch_size, max_length=cfg.max_length):
    a_win, b_win, tie = [], [], []
    
    for start_idx in tqdm(range(0, len(df), batch_size)):
        end_idx = min(start_idx + batch_size, len(df))
        tmp = df.iloc[start_idx:end_idx]
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        outputs = model(**inputs.to(device))
        proba = outputs.logits.softmax(-1).cpu()
        
        a_win.extend(proba[:, 0].tolist())
        b_win.extend(proba[:, 1].tolist())
        tie.extend(proba[:, 2].tolist())
    
    df["winner_model_a"] = a_win
    df["winner_model_b"] = b_win
    df["winner_tie"] = tie
    
    return df

st = time.time()

# sort by input length to fully leverage dynaminc padding
data = data.sort_values("length", ascending=False)
# the total #tokens in sub_1 and sub_2 should be more or less the same

results = inference(data, model, device)
print(results)
# result_df = pd.concat(list(results), axis=0)
result_df = results
proba = result_df[["winner_model_a", "winner_model_b", "winner_tie"]].values

print(f"elapsed time: {time.time() - st}")

st = time.time()

if cfg.tta:
    data = aug_data.sort_values("length", ascending=False)  # sort by input length to boost speed

    tta_result_df = inference(data, model, device)
    # recall TTA's order is flipped
    tta_proba = tta_result_df[["winner_model_b", "winner_model_a", "winner_tie"]].values 
    # average original result and TTA result.
    proba = (proba + tta_proba) / 2

print(f"elapsed time: {time.time() - st}")

result_df.loc[:, "winner_model_a"] = proba[:, 0]
result_df.loc[:, "winner_model_b"] = proba[:, 1]
result_df.loc[:, "winner_tie"] = proba[:, 2]
submission_df = result_df[["id", 'winner_model_a', 'winner_model_b', 'winner_tie']]
submission_df.to_csv('submission_final.csv', index=False)
# display(submission_df)
