# CITATIONS.md

## AI Assistance

### Task 1 — Financial AI Pipeline
- AI-ASSISTED: Claude (claude-sonnet-4-6), Prompt: 'Build fetch_ohlcv function with dynamic dates', Date: 2026-05-04
- AI-ASSISTED: Claude (claude-sonnet-4-6), Prompt: 'Compute RSI with Wilder smoothing from first principles', Date: 2026-05-04
- AI-ASSISTED: Claude (claude-sonnet-4-6), Prompt: 'Pydantic models for structured LLM output validation', Date: 2026-05-04
- AI-ASSISTED: Claude (claude-sonnet-4-6), Prompt: 'Generate styled HTML equity research report with matplotlib', Date: 2026-05-04

### Task 2 — QLoRA Fine-Tuning Pipeline
- AI-ASSISTED: Claude (claude-sonnet-4-6), Prompt: 'Design seed question taxonomy for financial Q&A dataset generation', Date: 2026-05-04
- AI-ASSISTED: Claude (claude-sonnet-4-6), Prompt: 'Build QLoRA training pipeline with BitsAndBytes 4-bit quantisation and PEFT LoRA config', Date: 2026-05-04
- AI-ASSISTED: Claude (claude-sonnet-4-6), Prompt: 'Implement ROUGE-L and BERTScore evaluation comparing base vs fine-tuned model', Date: 2026-05-04

## Libraries & Models Used

### Task 1
- yfinance: https://github.com/ranaroussi/yfinance
- Groq Python SDK: https://github.com/groq/groq-python
- Pydantic: https://github.com/pydantic/pydantic

### Task 2
- Hugging Face Transformers: https://github.com/huggingface/transformers
- PEFT (Parameter-Efficient Fine-Tuning): https://github.com/huggingface/peft
- BitsAndBytes: https://github.com/TimDettmers/bitsandbytes
- TRL (Transformer Reinforcement Learning): https://github.com/huggingface/trl
- Groq Python SDK (dataset generation): https://github.com/groq/groq-python
- evaluate: https://github.com/huggingface/evaluate
- bert_score: https://github.com/Tiiiger/bert_score

## Academic References

- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. https://arxiv.org/abs/2305.14314
- Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. https://arxiv.org/abs/2106.09685
- Microsoft Research. (2023). *Phi-2: The surprising power of small language models.* https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/
