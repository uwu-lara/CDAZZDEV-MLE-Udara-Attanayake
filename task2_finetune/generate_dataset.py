"""
Task 2A — Financial Q&A Dataset Generator
Generates 100+ training examples via Groq (teacher LLM),
saves as JSONL, computes diversity metrics, splits 80/10/10.
"""

import os
import json
import time
import random
import hashlib
from collections import Counter
from groq import Groq

# ── Config ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
TEACHER_MODEL = "llama3-70b-8192"
OUTPUT_DIR    = "task2_finetuning/data"
TARGET_COUNT  = 110   # generate a few extra to account for any filtered dupes

# ── Topic seeds (ensures diversity across financial sub-domains) ────────────
TOPIC_SEEDS = [
    # Technical Analysis
    ("What does it mean when a stock's RSI goes above 70?",
     "technical analysis, RSI, overbought"),
    ("Explain the MACD crossover signal and how traders use it.",
     "technical analysis, MACD"),
    ("What is a death cross and why do investors care about it?",
     "technical analysis, moving averages"),
    ("How do Bollinger Bands help identify volatility?",
     "technical analysis, Bollinger Bands"),
    ("What is the significance of a stock trading at its 200-day moving average?",
     "technical analysis, support levels"),

    # Fundamental Analysis
    ("How do you interpret a price-to-earnings ratio when analysing a stock?",
     "fundamental analysis, P/E ratio"),
    ("What does a high debt-to-equity ratio tell you about a company?",
     "fundamental analysis, leverage"),
    ("Explain the difference between revenue and earnings.",
     "fundamental analysis, income statement"),
    ("What is free cash flow and why is it important?",
     "fundamental analysis, cash flow"),
    ("How do you use the PEG ratio to find undervalued stocks?",
     "fundamental analysis, valuation"),
    ("What does EBITDA measure and when is it useful?",
     "fundamental analysis, EBITDA"),
    ("Explain book value and price-to-book ratio.",
     "fundamental analysis, book value"),

    # Macroeconomics
    ("How do rising interest rates affect stock prices?",
     "macroeconomics, interest rates"),
    ("What is the yield curve and what does an inverted yield curve signal?",
     "macroeconomics, yield curve, recession"),
    ("How does inflation impact different sectors of the stock market?",
     "macroeconomics, inflation, sector rotation"),
    ("What is quantitative easing and how does it affect markets?",
     "macroeconomics, monetary policy, QE"),
    ("Explain the relationship between the US dollar and commodity prices.",
     "macroeconomics, currency, commodities"),
    ("What does the Federal Reserve's dot plot indicate?",
     "macroeconomics, Fed, interest rate projections"),

    # Risk Management
    ("What is beta and how does it measure stock risk?",
     "risk, beta, volatility"),
    ("Explain the Sharpe ratio and what makes it a useful metric.",
     "risk, Sharpe ratio, risk-adjusted returns"),
    ("What is Value at Risk (VaR) and how is it calculated?",
     "risk, VaR, portfolio risk"),
    ("How does portfolio diversification reduce risk?",
     "risk, diversification, correlation"),
    ("What is a stop-loss order and when should you use one?",
     "risk, stop-loss, trading strategy"),

    # Earnings & Corporate Events
    ("What should investors look for in a company's earnings report?",
     "earnings, quarterly results"),
    ("How do stock splits affect share price and investor holdings?",
     "corporate events, stock split"),
    ("What is a dividend payout ratio and what does it tell investors?",
     "dividends, income investing"),
    ("Explain what happens during a company's IPO process.",
     "corporate events, IPO"),
    ("What is a share buyback and how does it affect earnings per share?",
     "corporate events, buybacks, EPS"),

    # Options & Derivatives
    ("What is the difference between a call option and a put option?",
     "options, derivatives"),
    ("Explain implied volatility and how it affects option pricing.",
     "options, implied volatility"),
    ("What is the Greeks in options trading — delta, gamma, theta?",
     "options, Greeks"),

    # Fixed Income
    ("How do bond prices move in relation to interest rates?",
     "bonds, fixed income, interest rates"),
    ("What is the difference between investment-grade and junk bonds?",
     "bonds, credit rating"),
    ("Explain duration in bond investing and why it matters.",
     "bonds, duration, interest rate risk"),

    # Market Structure
    ("What is market capitalisation and how is it calculated?",
     "market structure, market cap"),
    ("What is the difference between a bull market and a bear market?",
     "market structure, market cycles"),
    ("Explain the difference between growth stocks and value stocks.",
     "investing style, growth vs value"),
    ("What is short selling and what are the risks involved?",
     "short selling, market mechanics"),
    ("What are ETFs and how do they differ from mutual funds?",
     "ETFs, passive investing"),

    # Sector & Industry Analysis
    ("How do rising oil prices affect airline stocks?",
     "sector analysis, energy, airlines"),
    ("Why do technology stocks typically have higher P/E ratios?",
     "sector analysis, technology"),
    ("How does the housing market affect bank stocks?",
     "sector analysis, real estate, banks"),
    ("What drives pharmaceutical stock valuations?",
     "sector analysis, healthcare, pharma"),
]

SYSTEM_PROMPT = """You are an expert financial analyst and educator. 
Your job is to provide clear, accurate, educational answers to questions about financial markets, 
stock analysis, investing, and economics. 
Answers should be factual, well-structured, and suitable for intermediate-level investors.
Do not give personalised investment advice. Keep responses between 100–200 words."""


def generate_example(client: Groq, question: str, topic_hint: str) -> dict | None:
    """Call teacher LLM to generate one Q&A pair."""
    try:
        response = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        answer = response.choices[0].message.content.strip()
        if len(answer) < 50:          # skip suspiciously short answers
            return None
        return {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
            ],
            "topic": topic_hint,
        }
    except Exception as e:
        print(f"  ⚠ API error for '{question[:50]}': {e}")
        return None


def augment_question(base_question: str) -> str:
    """Create a slightly varied version of a seed question."""
    prefixes = [
        "Can you explain ", "In simple terms, ", "For a beginner investor, ",
        "From a portfolio manager's perspective, ", "Give me a brief overview of ",
        "Why is it important to understand ", "How would you describe ",
        "What exactly is meant by the term related to: ",
    ]
    suffixes = [
        " Give a practical example.",
        " Include a real-world scenario.",
        " Keep your answer concise.",
        " Highlight common misconceptions.",
        " Focus on the implications for retail investors.",
        "",
    ]
    p = random.choice(prefixes)
    s = random.choice(suffixes)
    # Simple augmentation: wrap with prefix/suffix
    q = base_question.rstrip("?.")
    return f"{p}{q[0].lower()}{q[1:]}?{s}"


def dedup(examples: list[dict]) -> list[dict]:
    """Remove near-duplicates by hashing the user question."""
    seen, out = set(), []
    for ex in examples:
        q = ex["messages"][1]["content"]
        h = hashlib.md5(q.lower().strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(ex)
    return out


def compute_diversity(examples: list[dict]) -> dict:
    """Compute basic diversity metrics."""
    topics   = [ex["topic"] for ex in examples]
    answers  = [ex["messages"][2]["content"] for ex in examples]
    lengths  = [len(a.split()) for a in answers]
    tc       = Counter(t.split(",")[0].strip() for t in topics)

    # Unique trigrams across all answers (lexical diversity proxy)
    all_words = " ".join(answers).lower().split()
    trigrams  = set(zip(all_words, all_words[1:], all_words[2:]))

    return {
        "total_examples":        len(examples),
        "unique_topics":         len(tc),
        "topic_distribution":    dict(tc.most_common()),
        "avg_answer_words":      round(sum(lengths) / len(lengths), 1),
        "min_answer_words":      min(lengths),
        "max_answer_words":      max(lengths),
        "unique_trigrams":       len(trigrams),
        "lexical_diversity":     round(len(trigrams) / max(sum(lengths), 1), 4),
    }


def split_and_save(examples: list[dict], out_dir: str):
    """Shuffle and write train / val / test JSONL files (80/10/10)."""
    os.makedirs(out_dir, exist_ok=True)
    random.shuffle(examples)
    n      = len(examples)
    n_val  = max(1, int(n * 0.10))
    n_test = max(1, int(n * 0.10))

    splits = {
        "train": examples[n_val + n_test :],
        "val":   examples[:n_val],
        "test":  examples[n_val : n_val + n_test],
    }

    for name, data in splits.items():
        path = os.path.join(out_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"  ✅ {name}.jsonl — {len(data)} examples → {path}")

    return splits


def main():
    print("=" * 60)
    print("Task 2A — Financial Q&A Dataset Generator")
    print("=" * 60)

    client = Groq(api_key=GROQ_API_KEY)
    examples: list[dict] = []

    total_seeds = len(TOPIC_SEEDS)
    print(f"\n📋 {total_seeds} seed topics loaded. Targeting {TARGET_COUNT} examples.\n")

    # ── Round 1: generate from every seed question ──────────────────────────
    print("── Round 1: Seed questions ─────────────────────────────────")
    for i, (question, topic) in enumerate(TOPIC_SEEDS, 1):
        print(f"  [{i:03d}/{total_seeds}] {question[:60]}...")
        ex = generate_example(client, question, topic)
        if ex:
            examples.append(ex)
        time.sleep(0.5)   # be polite to the API

    print(f"\n  Generated {len(examples)} from seeds.\n")

    # ── Round 2: augmented variants until we hit TARGET_COUNT ───────────────
    print("── Round 2: Augmented variants ─────────────────────────────")
    attempt = 0
    while len(examples) < TARGET_COUNT and attempt < TARGET_COUNT * 2:
        seed_q, topic = random.choice(TOPIC_SEEDS)
        aug_q         = augment_question(seed_q)
        attempt      += 1
        print(f"  [aug {attempt:03d}] {aug_q[:60]}...")
        ex = generate_example(client, aug_q, topic)
        if ex:
            examples.append(ex)
        time.sleep(0.4)

    print(f"\n  Total before dedup: {len(examples)}")

    # ── Dedup ────────────────────────────────────────────────────────────────
    examples = dedup(examples)
    print(f"  After dedup:        {len(examples)}\n")

    # ── Diversity metrics ────────────────────────────────────────────────────
    print("── Diversity Metrics ───────────────────────────────────────")
    metrics = compute_diversity(examples)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    metrics_path = os.path.join(OUTPUT_DIR, "diversity_metrics.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → {metrics_path}\n")

    # ── Split & save ─────────────────────────────────────────────────────────
    print("── Saving splits ───────────────────────────────────────────")
    splits = split_and_save(examples, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("✅ Dataset generation complete!")
    print(f"   Train: {len(splits['train'])}  |  Val: {len(splits['val'])}  |  Test: {len(splits['test'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
