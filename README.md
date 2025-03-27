# 🤖 Fine-Tuning Gemma-2-2B-it for Human Preference Prediction

This repository contains the code and experiments from my **Stanford CS224N (Winter 2025)** custom final project: fine-tuning the **Gemma-2-2B-it** large language model to align chatbot responses with **human preferences** in pairwise comparison settings. The project targets key biases in LLM outputs—**verbosity**, **position**, and **self-enhancement**—and proposes a fine-tuning pipeline using **LoRA**, **Test-Time Augmentation (TTA)**, and ablation testing with **PISSA initialization**.

📎 [Final Report PDF](https://github.com/BochengDai/Fine-tuning-Gemma-2-2-Bit/blob/main/CS224N__Project_Final_Report_2025_BochengDai.pdf)

## 📌 Overview

Given a user prompt and two chatbot responses, the task is to predict which one a human is more likely to prefer—or if the outcome is a tie. The model is trained on the **LMSYS Chatbot Arena dataset** and evaluated with a custom **bias-specific validation set**.

**Key contributions:**
- 🧠 **LoRA fine-tuning** on `Gemma-2-2B-it` (2.6B parameters)
- 🔁 **Test-Time Augmentation (TTA)** by swapping response order
- ⚙️ Optional **PISSA**-based SVD initialization for LoRA (shown to constrain model flexibility)
- 🧪 Custom **multi-layer classifier** head for preference scoring

---

## 🧠 Model Architecture

We use Hugging Face's `Gemma2ForSequenceClassification` with:
- A 3-layer classification head
- LoRA applied to: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`
- Full transformer layers trainable (no freezing)
- Optional LoRA EMA and R-Drop regularization
- Trained in bfloat16 with gradient accumulation + checkpointing

---

## 📊 Results

| Model Configuration                                | Accuracy | Log Loss |
|----------------------------------------------------|----------|----------|
| **LoRA + TTA (Best)**                              | **49.22%** | **1.0373** |
| LoRA + TTA + PISSA Initialization                  | 47.26%   | 1.0522   |
| LoRA only (no TTA, with PISSA)                     | 43.41%   | 1.0715   |
| Single-layer scoring head                          | 31.29%   | 1.2574   |
| XGBoost Benchmark                                  | 46.29%   | 1.0582   |
| Pretrained Gemma-2 (no tuning)                     | 29.78%   | 3.0237   |

---

## 🛠️ Setup

### 🔧 Environment & Model Download

To set up the environment and download the `Gemma-2-2B-it` model, simply run:

```bash
sh install.sh
python3 download_model.py
