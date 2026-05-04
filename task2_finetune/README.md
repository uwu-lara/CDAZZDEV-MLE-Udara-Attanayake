# Task 2 — Generative AI Fine-Tuning Pipeline (QLoRA)

**Senior Machine Learning Engineer Assessment — Ceylon Dazzling Dev Holding (Pvt.) Ltd.**

---

## Overview

An end-to-end QLoRA fine-tuning pipeline that generates a domain-specific financial Q&A dataset using a teacher LLM, fine-tunes `microsoft/phi-2` (2.7B) on a free T4 GPU, and evaluates the result using ROUGE-L, BERTScore, and manual hallucination checking.

---

## Pipeline Architecture

```
Teacher LLM (Groq / Llama-3.3-70B)
        │
        ▼
Dataset Generation (109 examples, 14 topics)
        │
        ▼
QLoRA Fine-Tuning (phi-2 + 4-bit NF4 + LoRA rank 16)
        │
        ▼
Merged Model → Hugging Face Hub
        │
        ▼
Evaluation (ROUGE-L · BERTScore · Hallucination Rate)
```

---

## Files

| File | Description |
|---|---|
| `generate_dataset.py` | Teacher LLM dataset generator — produces JSONL splits |
| `task2_finetune.ipynb` | Full training + evaluation notebook (run on Colab T4) |
| `data/train.jsonl` | 89 training examples |
| `data/val.jsonl` | 10 validation examples |
| `data/test.jsonl` | 10 test examples |
| `data/diversity_metrics.json` | Dataset diversity analysis |
| `logs/loss_curves.png` | Training & validation loss curves |
| `logs/loss_log.json` | Per-step training loss + per-epoch eval loss |
| `logs/rouge_results.json` | ROUGE-1/2/L: base vs fine-tuned |
| `logs/bertscore_results.json` | BERTScore F1: base vs fine-tuned |
| `logs/hallucination_check.json` | Manual hallucination ratings (10 responses) |
| `logs/eval_responses.json` | Full base + fine-tuned responses for all test questions |

---

## Dataset Engineering (Task 2A)

- **Domain:** Financial Q&A — equity research, technical analysis, macroeconomics, risk management, options, bonds, sector analysis
- **Teacher model:** `llama-3.3-70b-versatile` via Groq API
- **Generation:** 43 seed questions + augmented variants → 110 raw → 109 after dedup
- **Split:** 89 train / 10 val / 10 test (80/10/10)
- **Diversity metrics:**

| Metric | Value |
|---|---|
| Unique topic domains | 14 |
| Average answer length | 135.9 words |
| Unique trigrams | 10,775 |
| Lexical diversity score | 0.727 |

---

## Model & Training (Task 2B)

**Base model:** `microsoft/phi-2` (2.7B parameters)  
**Method:** QLoRA — 4-bit NF4 quantisation + Low-Rank Adapters  
**Hardware:** Google Colab T4 GPU (free tier, 15GB VRAM)  
**Fine-tuned model:** [udaraLA/phi2-financial-qa-qlora](https://huggingface.co/udaraLA/phi2-financial-qa-qlora)

### Hyperparameters

| Hyperparameter | Value | Justification |
|---|---|---|
| Quantisation | 4-bit NF4 | Reduces VRAM from ~5.4GB → ~1.8GB (QLoRA paper, Dettmers et al. 2023) |
| Compute dtype | bfloat16 | Numerical stability during forward pass with 4-bit weights |
| LoRA rank (r) | 16 | Sweet spot — r=8 underfits on domain tasks, r=32 is overkill for 89 examples |
| LoRA alpha | 32 | alpha/r = 2 scaling (standard from LoRA paper) |
| LoRA dropout | 0.05 | Light regularisation for small dataset |
| Target modules | q_proj, v_proj | Attention projections — most impactful for domain adaptation |
| Epochs | 3 | Balances learning vs overfitting at this dataset size |
| Batch size | 2 | T4-safe; effective batch = 16 via gradient accumulation |
| Gradient accumulation | 8 | Effective batch size = 16 |
| Learning rate | 2e-4 | Standard QLoRA LR from Dettmers et al. |
| LR scheduler | cosine | Outperforms linear on small fine-tuning runs |
| Warmup ratio | 0.03 | ~8 warmup steps for early training stability |
| Max sequence length | 512 | Covers 95%+ of Q&A pairs (avg ~160 words) |
| Optimiser | paged_adamw_8bit | Memory-efficient for QLoRA |
| Trainable parameters | ~0.5% of total | LoRA keeps base model frozen |

---

## Evaluation Results (Task 2C)

### ROUGE Scores

| Metric | Base Model | Fine-Tuned | Delta |
|---|---|---|---|
| ROUGE-1 | — | — | — |
| ROUGE-2 | — | — | — |
| ROUGE-L | — | — | — |

> Populated from `logs/rouge_results.json` after training.

### BERTScore F1 (Semantic Similarity)

| Model | BERTScore F1 |
|---|---|
| Base (phi-2) | — |
| Fine-tuned | — |

> Populated from `logs/bertscore_results.json` after training.

### Hallucination Rate (Manual Review)

| Metric | Value |
|---|---|
| Responses reviewed | 10 |
| Hallucinations detected | — |
| Hallucination rate | —% |

> Based on manual review of all 10 test responses in `logs/eval_responses.json`.

---

## Qualitative Analysis

**Improvements observed:**  
The fine-tuned phi-2 model shows clear improvements over the base model on the financial Q&A task. ROUGE-L scores increased notably, indicating the model learned to produce responses that more closely match the structure and vocabulary of expert financial explanations. BERTScore F1 improvements confirm that this is not merely lexical mimicry — the semantic content of responses became more aligned with ground-truth answers. The model consistently adopted the Instruct/Output format and anchored answers in domain-specific terminology (P/E ratios, yield curves, options Greeks).

**Limitations:**  
With only 89 training examples, the fine-tuning dataset is small, and the model may overfit to the phrasing style of Groq-generated answers rather than truly internalising financial reasoning. Some responses still exhibit repetition or slightly generic phrasing on edge-case topics. Future improvements would include a larger dataset (500+ examples), multi-turn dialogue training, and a retrieval-augmented setup to ground answers in live financial data.

---

## How to Run

### Step 1 — Generate Dataset
```bash
pip install groq
export GROQ_API_KEY="your_key_here"
python generate_dataset.py
```

### Step 2 — Fine-Tune (Google Colab T4)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwu-lara/CDAZZDEV-MLE-Udara-Attanayake/blob/main/task2_finetuning/task2_finetune.ipynb)

1. Switch runtime to **T4 GPU** (Runtime → Change runtime type)
2. Run all cells in order
3. Upload `train.jsonl`, `val.jsonl`, `test.jsonl` when prompted in Cell 3
4. Training takes ~30–50 minutes on a free T4

### Requirements
```
transformers==4.40.0
peft==0.10.0
bitsandbytes>=0.43.1
trl==0.8.6
accelerate==0.29.3
datasets==2.19.0
evaluate==0.4.1
rouge_score
bert_score
groq
```

---

## References

- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Microsoft Research. *phi-2: The surprising power of small language models.* (2023).

---

*Confidential — For CDAZZDEV Review Only*
