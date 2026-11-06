"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Tasks 1, 2, 4, 5 and bonuses are implemented; Task 3 is optional.
    2. Do NOT change function signatures.
    3. Work in starter-code/template.py; tests load this file by default.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from pathlib import Path
from itertools import chain
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ---------------------------------------------------------------------------
# Fixed lab prices per 1M INPUT & OUTPUT tokens (USD), not live billing rates.
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


def _require_key(name: str) -> str:
    key = os.getenv(name, "").strip()
    if not key:
        raise ValueError(f"Thiếu {name}. Hãy thiết lập trong .env hoặc biến môi trường.")
    return key


def _validate_request(prompt: str, max_tokens: int) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt không được để trống.")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens phải là số nguyên dương.")


def _api_request(fn: Callable[[], Any]) -> Any:
    """Retry transient HTTP/network errors; return permanent errors immediately."""
    def attempt():
        try:
            return fn(), None
        except Exception as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            transient = (
                status in (408, 429, 500, 502, 503, 504)
                or isinstance(exc, (TimeoutError, ConnectionError))
                or type(exc).__name__ in (
                    "APIConnectionError", "APITimeoutError", "ConnectError",
                    "ReadTimeout", "ConnectTimeout",
                )
            )
            if transient:
                raise
            return None, exc

    result, error = retry_with_backoff(attempt, max_retries=3, base_delay=1.0)
    if error is not None:
        raise error
    return result


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """

    _validate_request(prompt, max_tokens)
    key = _require_key("OPENAI_API_KEY")
    from openai import OpenAI

    client = OpenAI(api_key=key, timeout=30.0, max_retries=0)
    start = time.perf_counter()
    response = _api_request(lambda: client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    ))
    latency = time.perf_counter() - start

    response_text = response.choices[0].message.content or ""

    if response.usage is None:
        raise ValueError("OpenAI không trả về token usage; không thể tính chi phí.")
    usage = {
        "input_tokens": int(response.usage.prompt_tokens or 0),
        "output_tokens": int(response.usage.completion_tokens or 0),
    }
    return response_text, latency, usage



# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig

        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig

        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens'
        from the response metadata (e.g. response.usage_metadata).
    """
    _validate_request(prompt, max_tokens)
    key = _require_key("GEMINI_API_KEY")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=30_000))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    start = time.perf_counter()
    response = _api_request(lambda: client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    ))
    latency = time.perf_counter() - start
    response_text = response.text or ""
    if not response_text.strip():
        raise ValueError("Gemini không trả về nội dung; hãy kiểm tra bộ lọc an toàn hoặc giới hạn token.")
    if response.usage_metadata is None:
        raise ValueError("Gemini không trả về token usage; không thể tính chi phí.")
    usage = {
        "input_tokens": int(response.usage_metadata.prompt_token_count or 0),
        "output_tokens": int(response.usage_metadata.candidates_token_count or 0),
    }
    return response_text, latency, usage





# ---------------------------------------------------------------------------
# Task 3 — Optional; skipped for this submission (no Claude API key)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Optional reference implementation; not used by comparison or chatbot.
    The legacy default model must be updated before any real Claude API call.

    Set CLAUDE_MOCK=1 to return a clearly labelled demo response without an
    API key, network access, or the Anthropic SDK. Mock token usage is zero
    because no model inference occurs. This mode does not evaluate Claude.
    When mock mode is off, SDK client creation can be patched by unit tests;
    authentication for a real request is handled by the SDK.

    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    _validate_request(prompt, max_tokens)
    if os.getenv("CLAUDE_MOCK", "").strip().lower() in ("1", "true", "yes"):
        return (
            "[MOCK Claude] Đây là phản hồi giả lập để thử luồng xử lý; "
            "chưa gọi Claude API.",
            0.0,
            {"input_tokens": 0, "output_tokens": 0},
        )

    key = os.getenv("ANTHROPIC_API_KEY", "").strip() or None
    import anthropic

    client = anthropic.Anthropic(api_key=key, timeout=30.0, max_retries=0)
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    response_text = "".join(
        block.text for block in response.content
        if isinstance(getattr(block, "text", None), str)
    )
    usage = {
        "input_tokens": int(response.usage.input_tokens or 0),
        "output_tokens": int(response.usage.output_tokens or 0),
    }
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    openai_result = call_openai(prompt, model=OPENAI_MODEL)
    mini_result = call_openai(prompt, model=OPENAI_MINI_MODEL)
    gemini_result = call_gemini(prompt, model=GEMINI_MODEL)

    def model_summary(model_id: str, result: tuple[str, float, dict]) -> dict:
        text, latency, usage = result
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        rates = PRICING_1M_TOKENS[model_id]
        cost = (
            input_tokens * rates["input"]
            + output_tokens * rates["output"]
        ) / 1_000_000  # Prices are quoted per one million tokens.
        return {
            "response": text,
            "latency": latency,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    return {
        "gpt4o": model_summary(OPENAI_MODEL, openai_result),
        "gpt4o_mini": model_summary(OPENAI_MINI_MODEL, mini_result),
        "gemini_flash": model_summary(GEMINI_MODEL, gemini_result),
    }

# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    from google import genai

    from google.genai import types

    client = None
    history: list[dict] = []

    print("Gemini chatbot (nhập 'quit' hoặc 'exit' để thoát).")
    while True:
        try:
            prompt = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã kết thúc cuộc trò chuyện.")
            break

        if prompt.lower() in ("quit", "exit"):
            print("Đã kết thúc cuộc trò chuyện.")
            break
        if not prompt:
            continue

        if client is None:
            client = genai.Client(
                api_key=_require_key("GEMINI_API_KEY"),
                http_options=types.HttpOptions(timeout=30_000),
            )

        user_message = {"role": "user", "parts": [{"text": prompt}]}
        contents = history + [user_message]
        pieces: list[str] = []

        print("Gemini: ", end="", flush=True)
        try:
            def open_stream():
                # Retry only before the first chunk; never repeat printed text.
                iterator = iter(client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=contents,
                ))
                first = next(iterator, None)
                return chain([] if first is None else [first], iterator)

            stream = _api_request(open_stream)
            for chunk in stream:
                text_chunk = chunk.text or ""
                if text_chunk:
                    print(text_chunk, end="", flush=True)
                    pieces.append(text_chunk)
        except KeyboardInterrupt:
            print("\nĐã kết thúc cuộc trò chuyện.")
            break
        except Exception:
            print("\nKhông thể nhận phản hồi. Kiểm tra key, quota hoặc kết nối. Lượt này chưa được lưu vào lịch sử.")
            continue
        print()
        if not pieces:
            print("Không có nội dung trả về; lượt này chưa được lưu vào lịch sử.")
            continue

        history.extend([
            user_message,
            {"role": "model", "parts": [{"text": "".join(pieces)}]},
        ])
        history = history[-6:]  # Three turns = six messages.


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0 or base_delay < 0:
        raise ValueError("max_retries and base_delay must be non-negative")

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))

# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results: list[dict] = []
    for prompt in prompts:
        comparison = compare_models(prompt)
        results.append({"prompt": prompt, **comparison})
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    headers = ["Prompt", "Model", "Response (truncated)", "Latency",
               "Tokens (In/Out)", "Cost (USD)"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    def safe_cell(value: object) -> str:
        # Prevent response text from breaking Markdown table rows/columns.
        return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    model_names = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o Mini",
        "gemini_flash": "Gemini-Flash",
    }
    for item in results:
        prompt = safe_cell(item["prompt"])
        for key, display_name in model_names.items():
            data = item[key]
            response = " ".join(str(data["response"] or "").split())
            if len(response) > 50:
                response = response[:47] + "..."
            cells = [
                prompt,
                display_name,
                safe_cell(response),
                f'{data["latency"]:.2f}s',
                f'{data["input_tokens"]}/{data["output_tokens"]}',
                f'${data["cost"]:.6f}',
            ]
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
