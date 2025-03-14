from modelscope import snapshot_download

# Download the model to match config.py
model_dir = snapshot_download('LLM-Research/gemma-2-9b-it', cache_dir='./autodl-tmp')