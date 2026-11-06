"""
Day 1 — LLM API Foundation (solution)
AICB-P1: AI Practical Competency Program, Phase 1

Run from the repo root:
    pytest tests/test_solution.py -v                    # mocked, no API key needed
    python src/optional_llm_api/solution/solution.py    # real calls, needs OPENAI_API_KEY (.env)

Result keys / table headers still say "gpt4o" / "GPT-4o" / "Mini" (test contract):
read "gpt4o" as the large model (gpt-6.1-terra) and "mini" as the small model (gpt-6-luna).
"""

import os
import time
from typing import Any, Callable

try:  # local run: read OPENAI_API_KEY from .env at the repo root (never committed)
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Estimated costs per 1K OUTPUT tokens (USD) — OpenAI pricing page, 11/2026.
# ---------------------------------------------------------------------------
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-6.1-terra": 0.005,   # $5.00 / 1M output tokens
    "gpt-6-luna": 0.0005,     # $0.50 / 1M output tokens
}

OPENAI_MODEL = "gpt-6.1-terra"
OPENAI_MINI_MODEL = "gpt-6-luna"
HISTORY_TURNS = 3          # streaming chatbot keeps the last 3 user/assistant turns


def _client():
    from openai import OpenAI   # imported here so tests can patch openai.OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Task 1 — Call the large model (OPENAI_MODEL)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """Call Chat Completions; return (response_text, latency_seconds)."""
    client = _client()
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none",           # GPT-6 only accepts temperature/top_p with effort "none"
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_tokens,  # GPT-6 rejects max_tokens
    )
    latency = time.perf_counter() - start
    text = response.choices[0].message.content or ""
    return text, float(latency)


# ---------------------------------------------------------------------------
# Task 2 — Call the small model (OPENAI_MINI_MODEL)
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """Same as call_openai, with model=OPENAI_MINI_MODEL."""
    return call_openai(prompt, model=OPENAI_MINI_MODEL, temperature=temperature,
                       top_p=top_p, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Task 3 — Compare the large vs the small model
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """Send the same prompt to both models; return responses, latencies and a cost estimate."""
    big_text, big_latency = call_openai(prompt)
    mini_text, mini_latency = call_openai_mini(prompt)
    est_tokens = len(big_text.split()) / 0.75                      # ~0.75 words per token
    cost = est_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS[OPENAI_MODEL]
    return {
        "gpt4o_response": big_text,
        "mini_response": mini_text,
        "gpt4o_latency": big_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": cost,
    }


# ---------------------------------------------------------------------------
# Task 4 — Streaming chatbot with conversation history
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """Interactive terminal chatbot: streams tokens, keeps the last 3 turns, 'quit'/'exit' to stop."""
    client = None
    history: list[dict] = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if not user:
            continue

        client = client or _client()
        history.append({"role": "user", "content": user})
        stream = client.chat.completions.create(
            model=OPENAI_MINI_MODEL,
            messages=history,
            reasoning_effort="none",
            temperature=0.7,
            max_completion_tokens=512,
            stream=True,
        )
        print("Bot: ", end="", flush=True)
        parts = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            parts.append(delta)
            print(delta, end="", flush=True)
        print()
        history.append({"role": "assistant", "content": "".join(parts)})
        history = history[-2 * HISTORY_TURNS:]                      # last 3 turns = 6 messages


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """Call fn(); on exception retry up to max_retries times, sleeping base_delay * 2^attempt."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * 2 ** attempt)


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """compare_models for every prompt, each result tagged with its "prompt"."""
    return [{"prompt": p, **compare_models(p)} for p in prompts]


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """Text table: Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency."""
    def cut(text, n=40):
        text = " ".join(str(text).split())
        return text if len(text) <= n else text[: n - 3] + "..."

    headers = ["Prompt", "GPT-4o Response", "Mini Response", "GPT-4o Latency", "Mini Latency"]
    rows = [[cut(r.get("prompt", "")), cut(r.get("gpt4o_response", "")), cut(r.get("mini_response", "")),
             f"{r.get('gpt4o_latency', 0.0):.2f}s", f"{r.get('mini_latency', 0.0):.2f}s"] for r in results]
    widths = [max(len(h), *(len(row[i]) for row in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = lambda cells: " | ".join(c.ljust(w) for c, w in zip(cells, widths))
    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_prompt = "Explain the difference between temperature and top_p in one sentence."
    print("=== Comparing models ===")
    result = compare_models(test_prompt)
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Starting chatbot (type 'quit' to exit) ===")
    streaming_chatbot()
