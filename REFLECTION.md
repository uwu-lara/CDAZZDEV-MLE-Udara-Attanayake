# REFLECTION.md

## Architectural Decisions

**Task 1 — Financial AI Pipeline**

The pipeline is structured as a linear DAG: data ingestion → feature engineering → LLM reasoning → report generation. Each stage is encapsulated in a single pure function with typed inputs and outputs, making the system testable and easy to extend.

For indicator computation, I chose to implement all five from mathematical first principles rather than using TA-Lib. RSI uses Wilder's exponential smoothing (alpha=1/14) rather than a simple rolling average, which matches the original Wilder specification and produces more stable values on recent data.

For LLM integration, prompts are defined as module-level string constants and never constructed inline. This separation ensures prompt logic is versionable and reviewable independently of business logic. Pydantic v2 validators enforce schema correctness before any downstream code consumes LLM output — JSON decode errors and validation failures are caught, logged via Python's logging module, and surfaced without crashing the pipeline.

The Groq inference API was chosen for its generous free tier and low latency on Llama-3.3-70B. Temperature was set to 0.1 for sentiment (high consistency needed across headlines) and 0.2 for signal generation (slight creativity acceptable for qualitative reasoning).

---

**Task 2 — QLoRA Fine-Tuning Pipeline**

The fine-tuning pipeline follows a three-stage architecture: synthetic dataset generation via a teacher LLM, parameter-efficient fine-tuning using QLoRA, and quantitative evaluation against the base model.

For dataset generation, I used `llama-3.3-70b-versatile` as the teacher model rather than a smaller model, prioritising answer quality over generation speed. The 43 seed questions were manually curated across 10 financial sub-domains to ensure topic diversity before augmentation. Lexical diversity was measured via unique trigram density (0.727), providing a quantifiable signal that the dataset is not simply paraphrasing a narrow set of templates.

For fine-tuning, `microsoft/phi-2` (2.7B) was selected over larger models (e.g. Llama-3-7B) specifically because it fits comfortably on a free Colab T4 GPU in 4-bit while still being large enough to demonstrate meaningful domain adaptation. QLoRA with rank 16 was chosen based on the QLoRA paper's finding that r=16 provides a strong balance between expressiveness and training stability for domain-specific tasks. Targeting only `q_proj` and `v_proj` keeps the adapter footprint minimal — less than 0.5% of total parameters are trainable — while still achieving meaningful performance gains, consistent with the original LoRA paper's findings on attention projection adaptation.

The training used paged AdamW with 8-bit optimisation, which reduces optimiser state memory by ~75% compared to full-precision Adam, making it the correct choice for constrained GPU environments. Gradient accumulation over 8 steps simulates an effective batch size of 16 without exceeding VRAM limits.

---

## What I Would Improve With More Time

**Task 1**
- **Fallback news source**: yfinance news occasionally returns fewer than 10 headlines for less liquid tickers. A secondary RSS feed (e.g. Yahoo Finance RSS) would make the pipeline more robust.
- **Retry logic**: LLM API calls have no retry on transient failures. Exponential backoff with a maximum of 3 retries would make this production-ready.
- **Streaming report**: The HTML report is generated as a static snapshot. A live-updating dashboard using Streamlit would be more useful in a real analyst workflow.
- **Confidence calibration**: The sentiment confidence values come directly from the LLM and are not calibrated against ground truth. A small labelled dataset could be used to apply Platt scaling.

**Task 2**
- **Larger dataset**: 89 training examples is sufficient to demonstrate the pipeline but insufficient for robust generalisation. A production-grade version would use 500–2000 examples with human expert review of teacher LLM outputs.
- **DPO alignment**: After SFT, a Direct Preference Optimisation step using human-ranked response pairs would further reduce hallucination and improve factual grounding.
- **Retrieval-augmented generation**: Coupling the fine-tuned model with a vector store of real earnings reports and SEC filings would ground answers in verifiable sources rather than parametric knowledge alone.
- **Automated hallucination detection**: Manual hallucination review does not scale. An NLI-based entailment classifier (e.g. cross-encoder/nli-deberta-v3-base) could automate this for larger evaluation sets.

---

## Limitations Encountered

**Task 1**
- The Groq free tier has rate limits that require careful management of token usage across multiple API calls.
- yfinance P/E ratio data is occasionally stale or missing for certain tickers — the pipeline handles this gracefully by returning None.
- Bollinger Band signals are noisy on short timeframes; the current implementation uses a 20-day window which is standard but may generate false signals in high-volatility regimes.

**Task 2**
- `llama3-70b-8192` was decommissioned by Groq mid-pipeline, requiring a model swap to `llama-3.3-70b-versatile`. This highlighted the importance of pinning model versions in production pipelines.
- BitsAndBytes requires a specific CUDA-compiled binary that is not always available in fresh Colab environments — resolved by upgrading to the latest package version alongside a kernel restart.
- The free Colab T4 session has a ~4 hour time limit, which is sufficient for this pipeline but would require Colab Pro or a cloud GPU (e.g. AWS g4dn) for larger models or longer training runs.
- Merging the LoRA adapter back into the base model for HuggingFace upload requires reloading the base model in fp16, which temporarily doubles VRAM usage — careful memory management (del + empty_cache) between steps was necessary.
