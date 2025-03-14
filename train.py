# train.py

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.integrations import TensorBoardCallback

from transformers import GemmaTokenizerFast, AutoTokenizer
from data_processing import load_data, CustomTokenizer
from model import initialize_model
from trainer import compute_metrics, RDropTrainer
from config import Config

# Load config
config = Config()

# Initialize tokenizer
tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
# tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
# Load dataset
ds = load_data("./train.csv")

# Initialize model
model = initialize_model(config)

# Define tokenizer with max_length
encode = CustomTokenizer(tokenizer, max_length=config.max_length)
# import ipdb, ipdb.set_trace()
ds = ds.map(encode, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=config.output_dir,
    overwrite_output_dir=True,
    report_to="none",
    num_train_epochs=config.n_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    logging_steps=1,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=500,
    optim=config.optim_type,
    bf16=True,
    learning_rate=config.lr,
    warmup_steps=config.warmup_steps,
    gradient_checkpointing=True,
    metric_for_best_model="log_loss",
    deepspeed=None,
    save_only_model=True,
    lr_scheduler_type='linear',
    # report_to="tensorboard",  # 启用 TensorBoard
)

tensorboard_callback = TensorBoardCallback()


# Initialize Trainer
trainer = RDropTrainer(
    args=training_args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    callbacks=[TensorBoardCallback()],
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
