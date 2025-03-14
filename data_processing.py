# data_processing.py
import json
from datasets import Dataset
from transformers import PreTrainedTokenizerBase, GemmaTokenizerFast

class CustomTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch: dict) -> dict:
        prompt = ["<prompt>: " + self.process_text(t) for t in batch["prompt"]]
        response_a = ["\n\n<response_a>: " + self.process_text(t) for t in batch["response_a"]]
        response_b = ["\n\n<response_b>: " + self.process_text(t) for t in batch["response_b"]]
        
        texts = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        # try:
        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        assert all(isinstance(t, str) for t in texts), "texts need to be a list of strings"
        # except Exception as exp:
            # print(texts)
        labels = []
        for a_win, b_win in zip(batch["winner_model_a"], batch["winner_model_b"]):
            if a_win:
                labels.append(0)
            elif b_win:
                labels.append(1)
            else:
                labels.append(2)
                
        return {**tokenized, "labels": labels}
    
    # @staticmethod
    # def process_text(text: str) -> str:
    #     return " ".join(eval(text, {"null": ""}))
    @staticmethod
    def process_text(text: str) -> str:
        try:
            return " ".join(json.loads(text))  # 确保解析后是字符串
        except:
            return str(text)  # 确保不会返回 None

def load_data(file_path: str) -> Dataset:
    return Dataset.from_csv(file_path)
