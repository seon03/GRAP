"""
generate_responses.py
=====================
Generate LLM responses for the GRAP gender-bias benchmark.

Supported models:
  - LLaMA 3.3 70B     : via a local vllm server (OpenAI-compatible API)
  - Gemini 2.5 Flash  : via OpenRouter (OpenAI-compatible API)
  - Claude Sonnet 4.5 : via OpenRouter (OpenAI-compatible API)
  - GPT-5.4-mini      : via OpenRouter (OpenAI-compatible API)

Responses are saved to  data/response/response_{model}.csv
Each CSV has columns: model, prompt_a, response_a, prompt_b, response_b

Usage:
  # LLaMA only  (requires a running vllm server)
  python src/generate_responses.py --model llama

  # Gemini only
  python src/generate_responses.py --model gemini

  # Claude only
  python src/generate_responses.py --model claude

  # GPT-5.4-mini only
  python src/generate_responses.py --model gpt

  # All models
  python src/generate_responses.py --model all

  # Quick test (first 10 rows, saved to data/response/test/)
  python src/generate_responses.py --model llama --limit 10

Options:
  --model   llama | gemini | claude | gpt | all  (default: all)
  --limit   N                                    number of rows to generate (default: all 1000)
  --pool    N                                    max concurrent API calls  (default: 32)

Environment variables (set in .env):
  VLLM_PORT          port for the local vllm server  (default: 8000)
  OPENROUTER_API_KEY API key for OpenRouter
"""

import argparse
import asyncio
import csv
import os
import time

from openai import AsyncOpenAI

# ── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_CSV = os.path.join(REPO_ROOT, "data", "prompt.csv")
RESP_DIR   = os.path.join(REPO_ROOT, "data", "response")

# ── Model constants ────────────────────────────────────────────────────────────

LLAMA_MODEL_NAME      = "llama-3.3-70b-instruct"
GEMINI_MODEL_NAME     = "gemini-2.5-flash"
CLAUDE_MODEL_NAME     = "claude-sonnet-4-5"
GPT_MODEL_NAME        = "gpt-5.4-mini"
GEMINI_OPENROUTER_ID  = "google/gemini-2.5-flash"
CLAUDE_OPENROUTER_ID  = "anthropic/claude-sonnet-4-5"
GPT_OPENROUTER_ID     = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL   = "https://openrouter.ai/api/v1"
MAX_TOKENS            = 1024


# ── Common helpers ─────────────────────────────────────────────────────────────

def load_prompts() -> list[tuple[str, str]]:
    """Load (prompt_a, prompt_b) pairs from data/prompt.csv."""
    with open(PROMPT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(r["prompt_a"], r["prompt_b"]) for r in rows]


def _extract_answer(response) -> str:
    """Return the final answer text, stripping <think>…</think> if present."""
    text = response.choices[0].message.content or ""
    if "</think>" in text:
        text = text.split("</think>")[-1].replace("<think>", "").strip()
    return text.strip()


def _extract_answer_with_thinking(response) -> tuple[str, str]:
    """Return (thinking, answer). Supports message.reasoning field and <think> tags."""
    content = response.choices[0].message.content or ""
    thinking = getattr(response.choices[0].message, "reasoning", None) or ""
    if not thinking and "</think>" in content:
        parts = content.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        content  = parts[1].strip()
    return thinking.strip(), content.strip()


def _output_path(model_name: str, output_dir: str) -> str:
    return os.path.join(output_dir, f"response_{model_name}.csv")


# ── LLaMA via vllm ────────────────────────────────────────────────────────────

async def _call_vllm(
    client: AsyncOpenAI,
    model_id: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                )
            return _extract_answer(resp)
        except Exception as e:
            err = str(e).lower()
            if "context" in err or "length" in err:
                return f"[ERROR] context length: {e}"
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            return f"[ERROR] {e}"
    return "[ERROR] max retries exceeded"


async def _generate_llama_async(
    prompts: list[tuple[str, str]],
    output_path: str,
    port: str,
    pool_size: int,
    done_indices: set,
) -> None:
    print(f"[LLaMA] Connecting to vllm server at http://localhost:{port}/v1", flush=True)
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
        timeout=300.0,
        max_retries=0,
    )
    try:
        models  = await client.models.list()
        model_id = models.data[0].id
        print(f"[LLaMA] Model on server: {model_id}  (port={port})", flush=True)
    except Exception as e:
        print(f"[LLaMA] Connection failed: {e}", flush=True)
        print(f"[LLaMA] Check that VLLM_PORT={port} is correct.", flush=True)
        return

    fieldnames = ["model", "prompt_a", "response_a", "prompt_b", "response_b"]

    existing_rows: dict[int, dict] = {}
    if os.path.isfile(output_path) and done_indices:
        with open(output_path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i in done_indices:
                    existing_rows[i] = row

    semaphore = asyncio.Semaphore(pool_size)
    total     = len(prompts)
    lock      = asyncio.Lock()
    results   = dict(existing_rows)
    completed = len(done_indices)
    t0        = time.time()

    async def _process(i: int, prompt_a: str, prompt_b: str) -> None:
        nonlocal completed
        print(f"[LLaMA] ({i+1}/{total}) generating prompt_a...", flush=True)
        response_a = await _call_vllm(client, model_id, prompt_a, semaphore)
        print(f"[LLaMA] ({i+1}/{total}) prompt_a done: {response_a[:80]!r}", flush=True)

        print(f"[LLaMA] ({i+1}/{total}) generating prompt_b...", flush=True)
        response_b = await _call_vllm(client, model_id, prompt_b, semaphore)
        print(f"[LLaMA] ({i+1}/{total}) prompt_b done: {response_b[:80]!r}", flush=True)

        async with lock:
            results[i] = {
                "model":      LLAMA_MODEL_NAME,
                "prompt_a":   prompt_a,
                "response_a": response_a,
                "prompt_b":   prompt_b,
                "response_b": response_b,
            }
            completed += 1
            elapsed     = time.time() - t0
            done_so_far = completed - len(done_indices)
            rate        = done_so_far / elapsed if elapsed > 0 else 0
            eta_m       = (total - completed) / rate / 60 if rate > 0 else float("inf")
            print(f"[LLaMA] {completed}/{total}  {rate:.2f} row/s  ETA {eta_m:.1f}min", flush=True)

    tasks = [_process(i, pa, pb) for i, (pa, pb) in enumerate(prompts) if i not in done_indices]
    await asyncio.gather(*tasks, return_exceptions=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(total):
            if i in results:
                writer.writerow(results[i])
    print(f"[LLaMA] Saved: {output_path}", flush=True)


def generate_llama(prompts: list, output_dir: str, port: str, pool_size: int) -> None:
    output_path = _output_path(LLAMA_MODEL_NAME, output_dir)
    done_indices: set = set()
    if os.path.isfile(output_path):
        with open(output_path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if "[ERROR]" not in row.get("response_a", "") and "[ERROR]" not in row.get("response_b", ""):
                    done_indices.add(i)
        print(f"[LLaMA] Resume: {len(done_indices)} rows already done (skipping)", flush=True)
    asyncio.run(_generate_llama_async(prompts, output_path, port, pool_size, done_indices))


# ── Gemini via OpenRouter ──────────────────────────────────────────────────────

async def _generate_gemini_async(
    prompts: list[tuple[str, str]],
    output_path: str,
    api_key: str,
    pool_size: int,
    done_indices: set,
) -> None:
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=120.0,
        max_retries=0,
    )
    fieldnames = ["model", "prompt_a", "thinking_a", "response_a", "prompt_b", "thinking_b", "response_b"]

    existing_rows: dict[int, dict] = {}
    if os.path.isfile(output_path) and done_indices:
        with open(output_path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i in done_indices:
                    existing_rows[i] = row

    semaphore = asyncio.Semaphore(pool_size)
    total     = len(prompts)
    lock      = asyncio.Lock()
    results   = dict(existing_rows)
    completed = len(done_indices)
    t0        = time.time()

    async def _call_gemini(prompt: str) -> tuple[str, str]:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(3):
            try:
                async with semaphore:
                    resp = await client.chat.completions.create(
                        model=GEMINI_OPENROUTER_ID,
                        messages=messages,
                        max_tokens=MAX_TOKENS,
                        temperature=0.7,
                    )
                return _extract_answer_with_thinking(resp)
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                return "", f"[ERROR] {e}"
        return "", "[ERROR] max retries exceeded"

    async def _process(i: int, prompt_a: str, prompt_b: str) -> None:
        nonlocal completed
        print(f"[Gemini] ({i+1}/{total}) generating prompt_a...", flush=True)
        thinking_a, response_a = await _call_gemini(prompt_a)
        print(f"[Gemini] ({i+1}/{total}) prompt_a done (thinking={len(thinking_a)}ch): {response_a[:80]!r}", flush=True)

        print(f"[Gemini] ({i+1}/{total}) generating prompt_b...", flush=True)
        thinking_b, response_b = await _call_gemini(prompt_b)
        print(f"[Gemini] ({i+1}/{total}) prompt_b done (thinking={len(thinking_b)}ch): {response_b[:80]!r}", flush=True)

        async with lock:
            results[i] = {
                "model":      GEMINI_MODEL_NAME,
                "prompt_a":   prompt_a,
                "thinking_a": thinking_a,
                "response_a": response_a,
                "prompt_b":   prompt_b,
                "thinking_b": thinking_b,
                "response_b": response_b,
            }
            completed += 1
            elapsed     = time.time() - t0
            done_so_far = completed - len(done_indices)
            rate        = done_so_far / elapsed if elapsed > 0 else 0
            eta_m       = (total - completed) / rate / 60 if rate > 0 else float("inf")
            print(f"[Gemini] {completed}/{total}  {rate:.2f} row/s  ETA {eta_m:.1f}min", flush=True)

    tasks = [_process(i, pa, pb) for i, (pa, pb) in enumerate(prompts) if i not in done_indices]
    await asyncio.gather(*tasks, return_exceptions=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(total):
            if i in results:
                writer.writerow(results[i])
    print(f"[Gemini] Saved: {output_path}", flush=True)


def generate_gemini(prompts: list, output_dir: str, pool_size: int) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set the OPENROUTER_API_KEY environment variable.")
    output_path = _output_path(GEMINI_MODEL_NAME, output_dir)
    done_indices: set = set()
    if os.path.isfile(output_path):
        with open(output_path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if "[ERROR]" not in row.get("response_a", "") and "[ERROR]" not in row.get("response_b", ""):
                    done_indices.add(i)
        print(f"[Gemini] Resume: {len(done_indices)} rows already done (skipping)", flush=True)
    asyncio.run(_generate_gemini_async(prompts, output_path, api_key, pool_size, done_indices))


# ── Claude via OpenRouter ──────────────────────────────────────────────────────

async def _generate_claude_async(
    prompts: list[tuple[str, str]],
    output_path: str,
    api_key: str,
    pool_size: int,
    done_indices: set,
) -> None:
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=120.0,
        max_retries=0,
    )
    fieldnames = ["model", "prompt_a", "response_a", "prompt_b", "response_b"]

    existing_rows: dict[int, dict] = {}
    if os.path.isfile(output_path) and done_indices:
        with open(output_path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i in done_indices:
                    existing_rows[i] = row

    semaphore = asyncio.Semaphore(pool_size)
    total     = len(prompts)
    lock      = asyncio.Lock()
    results   = dict(existing_rows)
    completed = len(done_indices)
    t0        = time.time()

    async def _call_claude(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(3):
            try:
                async with semaphore:
                    resp = await client.chat.completions.create(
                        model=CLAUDE_OPENROUTER_ID,
                        messages=messages,
                        max_tokens=MAX_TOKENS,
                        temperature=0.7,
                    )
                return _extract_answer(resp)
            except Exception as e:
                err = str(e).lower()
                if "rate_limit" in err or "429" in err:
                    wait = 2 ** (attempt + 2)
                    await asyncio.sleep(wait)
                    continue
                if attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                return f"[ERROR] {e}"
        return "[ERROR] max retries exceeded"

    async def _process(i: int, prompt_a: str, prompt_b: str) -> None:
        nonlocal completed
        print(f"[Claude] ({i+1}/{total}) generating prompt_a...", flush=True)
        response_a = await _call_claude(prompt_a)
        print(f"[Claude] ({i+1}/{total}) prompt_a done: {response_a[:80]!r}", flush=True)

        print(f"[Claude] ({i+1}/{total}) generating prompt_b...", flush=True)
        response_b = await _call_claude(prompt_b)
        print(f"[Claude] ({i+1}/{total}) prompt_b done: {response_b[:80]!r}", flush=True)

        async with lock:
            results[i] = {
                "model":      CLAUDE_MODEL_NAME,
                "prompt_a":   prompt_a,
                "response_a": response_a,
                "prompt_b":   prompt_b,
                "response_b": response_b,
            }
            completed += 1
            elapsed     = time.time() - t0
            done_so_far = completed - len(done_indices)
            rate        = done_so_far / elapsed if elapsed > 0 else 0
            eta_m       = (total - completed) / rate / 60 if rate > 0 else float("inf")
            print(f"[Claude] {completed}/{total}  {rate:.2f} row/s  ETA {eta_m:.1f}min", flush=True)

    tasks = [_process(i, pa, pb) for i, (pa, pb) in enumerate(prompts) if i not in done_indices]
    await asyncio.gather(*tasks, return_exceptions=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(total):
            if i in results:
                writer.writerow(results[i])
    print(f"[Claude] Saved: {output_path}", flush=True)


def generate_claude(prompts: list, output_dir: str, pool_size: int) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set the OPENROUTER_API_KEY environment variable.")
    output_path = _output_path(CLAUDE_MODEL_NAME, output_dir)
    done_indices: set = set()
    if os.path.isfile(output_path):
        with open(output_path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if "[ERROR]" not in row.get("response_a", "") and "[ERROR]" not in row.get("response_b", ""):
                    done_indices.add(i)
        print(f"[Claude] Resume: {len(done_indices)} rows already done (skipping)", flush=True)
    asyncio.run(_generate_claude_async(prompts, output_path, api_key, pool_size, done_indices))


# ── GPT-5.4-mini via OpenRouter ───────────────────────────────────────────────

async def _generate_gpt_async(
    prompts: list[tuple[str, str]],
    output_path: str,
    api_key: str,
    pool_size: int,
    done_indices: set,
) -> None:
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=120.0,
        max_retries=0,
    )
    fieldnames = ["model", "prompt_a", "response_a", "prompt_b", "response_b"]

    existing_rows: dict[int, dict] = {}
    if os.path.isfile(output_path) and done_indices:
        with open(output_path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i in done_indices:
                    existing_rows[i] = row

    semaphore = asyncio.Semaphore(pool_size)
    total     = len(prompts)
    lock      = asyncio.Lock()
    results   = dict(existing_rows)
    completed = len(done_indices)
    t0        = time.time()

    async def _call_gpt(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(3):
            try:
                async with semaphore:
                    resp = await client.chat.completions.create(
                        model=GPT_OPENROUTER_ID,
                        messages=messages,
                        max_tokens=MAX_TOKENS,
                        temperature=0.7,
                    )
                return _extract_answer(resp)
            except Exception as e:
                err = str(e).lower()
                if "rate_limit" in err or "429" in err:
                    wait = 2 ** (attempt + 2)
                    await asyncio.sleep(wait)
                    continue
                if attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                return f"[ERROR] {e}"
        return "[ERROR] max retries exceeded"

    async def _process(i: int, prompt_a: str, prompt_b: str) -> None:
        nonlocal completed
        print(f"[GPT] ({i+1}/{total}) generating prompt_a...", flush=True)
        response_a = await _call_gpt(prompt_a)
        print(f"[GPT] ({i+1}/{total}) prompt_a done: {response_a[:80]!r}", flush=True)

        print(f"[GPT] ({i+1}/{total}) generating prompt_b...", flush=True)
        response_b = await _call_gpt(prompt_b)
        print(f"[GPT] ({i+1}/{total}) prompt_b done: {response_b[:80]!r}", flush=True)

        async with lock:
            results[i] = {
                "model":      GPT_MODEL_NAME,
                "prompt_a":   prompt_a,
                "response_a": response_a,
                "prompt_b":   prompt_b,
                "response_b": response_b,
            }
            completed += 1
            elapsed     = time.time() - t0
            done_so_far = completed - len(done_indices)
            rate        = done_so_far / elapsed if elapsed > 0 else 0
            eta_m       = (total - completed) / rate / 60 if rate > 0 else float("inf")
            print(f"[GPT] {completed}/{total}  {rate:.2f} row/s  ETA {eta_m:.1f}min", flush=True)

    tasks = [_process(i, pa, pb) for i, (pa, pb) in enumerate(prompts) if i not in done_indices]
    await asyncio.gather(*tasks, return_exceptions=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(total):
            if i in results:
                writer.writerow(results[i])
    print(f"[GPT] Saved: {output_path}", flush=True)


def generate_gpt(prompts: list, output_dir: str, pool_size: int) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set the OPENROUTER_API_KEY environment variable.")
    output_path = _output_path(GPT_MODEL_NAME, output_dir)
    done_indices: set = set()
    if os.path.isfile(output_path):
        with open(output_path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if "[ERROR]" not in row.get("response_a", "") and "[ERROR]" not in row.get("response_b", ""):
                    done_indices.add(i)
        print(f"[GPT] Resume: {len(done_indices)} rows already done (skipping)", flush=True)
    asyncio.run(_generate_gpt_async(prompts, output_path, api_key, pool_size, done_indices))


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate LLM responses for GRAPH benchmark")
    parser.add_argument("--model",  choices=["llama", "gemini", "claude", "gpt", "all"], default="all",
                        help="Which model(s) to generate responses for (default: all)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Number of rows to generate (default: all 1000)")
    parser.add_argument("--pool",   type=int, default=32,
                        help="Max concurrent API calls (default: 32)")
    args = parser.parse_args()

    port = os.environ.get("VLLM_PORT", "8000")

    prompts = load_prompts()
    if args.limit:
        prompts = prompts[:args.limit]
    print(f"Loaded {len(prompts)} prompt pairs from {PROMPT_CSV}", flush=True)

    output_dir = os.path.join(RESP_DIR, "test") if args.limit else RESP_DIR
    os.makedirs(output_dir, exist_ok=True)

    if args.model in ("llama", "all"):
        generate_llama(prompts, output_dir, port, args.pool)

    if args.model in ("gemini", "all"):
        generate_gemini(prompts, output_dir, args.pool)

    if args.model in ("claude", "all"):
        generate_claude(prompts, output_dir, args.pool)

    if args.model in ("gpt", "all"):
        generate_gpt(prompts, output_dir, args.pool)


if __name__ == "__main__":
    main()
