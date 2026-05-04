# CDAZZDEV-MLE- Udara Attanayake

Senior Machine Learning Engineer — Technical Assessment  
Ceylon Dazzling Dev Holding (Pvt.) Ltd.

---

## Tasks Completed

| Task | Domain | Status |
|------|--------|--------|
| Task 1 | Financial AI — LLM-Powered Equity Research Assistant | ✅ Completed |
| Task 3 | Agentic Workflows — Multi-Agent Financial Research System | ✅ Completed |
---

## Task 1 — Financial AI Pipeline

End-to-end automated equity research assistant that ingests real financial market data,
computes technical indicators from first principles, applies LLM-based sentiment analysis
and signal reasoning, and produces a styled HTML research brief.

**Features:**
- 2+ years of daily OHLCV data via yfinance
- 5 technical indicators computed from scratch (no TA-Lib): SMA 50/200, RSI (14) with Wilder's smoothing, MACD (12,26,9), Bollinger Bands (20, 2σ)
- 10 recent news headlines with per-headline LLM sentiment analysis
- Structured Buy/Hold/Sell signal with multi-indicator reasoning
- Pydantic v2 validation on all LLM outputs
- Bonus: Styled dark-themed HTML report with embedded matplotlib chart

**Stack:** Python, yfinance, pandas, numpy, matplotlib, Groq API (Llama-3.3-70B), Pydantic v2

---

## How to Run

1. Open the notebook in Google Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1W3LoSGvp-XYdUGVcfxfMSmTGJzN2DTv8?usp=sharing)

2. Add your Groq API key to Colab Secrets as `GROQ_API_KEY`

3. Run all cells in order — outputs are pre-populated in the submitted notebook

---

## Notes
- No API keys or credentials stored anywhere in this repository
- All notebook cell outputs are visible and were not cleared before submission
- All AI assistance documented in CITATIONS.md

---

*Confidential — For CDAZZDEV Review Only*
