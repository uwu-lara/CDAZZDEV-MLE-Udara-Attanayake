# REFLECTION.md

## Architectural Decisions

**Task 1 — Financial AI Pipeline**

The pipeline is structured as a linear DAG: data ingestion → feature engineering → LLM reasoning → report generation. Each stage is encapsulated in a single pure function with typed inputs and outputs, making the system testable and easy to extend.

For indicator computation, I chose to implement all five from mathematical first principles rather than using TA-Lib. RSI uses Wilder's exponential smoothing (alpha=1/14) rather than a simple rolling average, which matches the original Wilder specification and produces more stable values on recent data.

For LLM integration, prompts are defined as module-level string constants and never constructed inline. This separation ensures prompt logic is versionable and reviewable independently of business logic. Pydantic v2 validators enforce schema correctness before any downstream code consumes LLM output — JSON decode errors and validation failures are caught, logged via Python's logging module, and surfaced without crashing the pipeline.

The Groq inference API was chosen for its generous free tier and low latency on Llama-3.3-70B. Temperature was set to 0.1 for sentiment (high consistency needed across headlines) and 0.2 for signal generation (slight creativity acceptable for qualitative reasoning).

## What I Would Improve With More Time

- **Fallback news source**: yfinance news occasionally returns fewer than 10 headlines for less liquid tickers. A secondary RSS feed (e.g. Yahoo Finance RSS) would make the pipeline more robust.
- **Retry logic**: LLM API calls have no retry on transient failures. Exponential backoff with a maximum of 3 retries would make this production-ready.
- **Streaming report**: The HTML report is generated as a static snapshot. A live-updating dashboard using Streamlit would be more useful in a real analyst workflow.
- **Confidence calibration**: The sentiment confidence values come directly from the LLM and are not calibrated against ground truth. A small labelled dataset could be used to apply Platt scaling.

## Limitations Encountered

- The Groq free tier has rate limits that require careful management of token usage across multiple API calls.
- yfinance P/E ratio data is occasionally stale or missing for certain tickers — the pipeline handles this gracefully by returning None.
- Bollinger Band signals are noisy on short timeframes; the current implementation uses a 20-day window which is standard but may generate false signals in high-volatility regimes.