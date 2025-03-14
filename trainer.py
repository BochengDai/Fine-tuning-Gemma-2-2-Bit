# trainer.py
# import ipdb
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.integrations import TensorBoardCallback
from transformers.trainer_callback import TrainerCallback
from sklearn.metrics import log_loss, accuracy_score
import torch.nn.functional as F

# Metric computation
def compute_metrics(eval_preds):
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

# LoRA EMA (Exponential Moving Average)
class LoRAEMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# Trainer class for custom loss and EMA
class RDropTrainer(Trainer):
    def __init__(self, *args, alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        # First forward pass
        outputs1 = model(**inputs)
        loss1 = outputs1.loss
        logits1 = outputs1.logits

        # Second forward pass
        outputs2 = model(**inputs)
        loss2 = outputs2.loss
        logits2 = outputs2.logits

        # KL divergence loss (R-Drop)
        kl_loss = compute_kl_loss(logits1, logits2)

        # Combine losses
        loss = (loss1 + loss2) / 2 + self.alpha * kl_loss
        return (loss, outputs1) if return_outputs else loss
