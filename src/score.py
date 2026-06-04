"""
score.py
========
Score LLM responses using the GRAPH rubric (20 dimensions).

The judge LLM (GPT-4o-mini) evaluates each prompt pair on every rubric
dimension in a separate API call, returning score_a and score_b in [0, 1].

Output JSONL format (one record per row):
  {
    "row_index": int,
    "model": str,
    "prompt_a": str, "response_a": str,
    "prompt_b": str, "response_b": str,
    "scores": {
      "favorability":    {"score_a": 0.75, "score_b": 0.5},
      ...                (20 dimensions total)
    }
  }

Usage:
  # Score a specific response CSV (full evaluation — question + response both shown to judge)
  INPUT=data/response/response_gpt-5.4-mini.csv \
    OPENROUTER_API_KEY=<key> python src/score.py

  # Response-only evaluation (question hidden from judge)
  INPUT=data/response/response_gpt-5.4-mini.csv \
    OPENROUTER_API_KEY=<key> python src/score.py --response_only

  # Use OpenAI API directly instead of OpenRouter
  INPUT=data/response/response_gpt-5.4-mini.csv \
    OPENAI_API_KEY=<key> python src/score.py

Environment variables:
  INPUT       path to response CSV  (default: data/response/response_gpt-5.4-mini.csv)
  LIMIT       number of rows to score  (default: 1000)
  POOL_SIZE   max concurrent API calls (default: 32)
  MODEL       judge model ID            (default: openai/gpt-4o-mini via OpenRouter,
                                                   gpt-4o-mini via OpenAI directly)
  OPENROUTER_API_KEY  preferred; uses https://openrouter.ai/api/v1
  OPENAI_API_KEY      fallback; uses OpenAI API directly

Output location:
  Full evaluation    : results/scores/full/score_{input_basename}.jsonl
  Response-only      : results/scores/response_only/score_{input_basename}_response_only.jsonl

Resume: already-completed row indices are skipped automatically.

Rate-limit note:
  POOL_SIZE defaults to 32. If 429 errors occur frequently, set POOL_SIZE=16.
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import time
from typing import Optional

from openai import AsyncOpenAI

logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUBRIC_PATH = os.path.join(REPO_ROOT, "rubric", "bias_rubric.json")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


# ── Rubric loader ──────────────────────────────────────────────────────────────

def load_rubric(rubric_path: str) -> tuple[str, str, list[dict]]:
    """Return (system_prompt, user_prompt_template, dimensions) from rubric JSON."""
    with open(rubric_path, encoding="utf-8") as f:
        rubric = json.load(f)
    return (
        rubric["system_prompt"],
        rubric["user_prompt_template"],
        rubric["dimensions"],
    )


# ── Single-dimension evaluator ─────────────────────────────────────────────────

async def _eval_one_dimension(
    dim: dict,
    prompt_a: str,
    response_a: str,
    prompt_b: str,
    response_b: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    system_prompt: str,
    user_prompt_template: str,
    row_index: int,
    max_retries: int = 4,
) -> Optional[dict]:
    user_content = user_prompt_template.format(
        aspect=dim["aspect"],
        aspect_definition=dim["aspect_definition"],
        question_a=prompt_a,
        answer_a=response_a,
        question_b=prompt_b,
        answer_b=response_b,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=64,
                    response_format={"type": "json_object"},
                )
            result = json.loads(resp.choices[0].message.content)
            return {
                "score_a": float(result.get("score_a", -1)),
                "score_b": float(result.get("score_b", -1)),
            }
        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                wait = 2 ** (attempt + 2)
                logging.warning(
                    "[SCORE] rate limit row=%d dim=%s — waiting %ds (attempt %d/%d)",
                    row_index, dim["key"], wait, attempt + 1, max_retries,
                )
                await asyncio.sleep(wait)
                continue
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            logging.warning(
                "[SCORE] failed row=%d dim=%s after %d attempts: %s",
                row_index, dim["key"], max_retries, exc,
            )
            return None

    return None


# ── Row-level evaluator ────────────────────────────────────────────────────────

async def _eval_row(
    row: dict,
    dimensions: list[dict],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    system_prompt: str,
    user_prompt_template: str,
) -> dict:
    row_index  = row["row_index"]
    prompt_a   = row["prompt_a"]
    response_a = row["response_a"]
    prompt_b   = row["prompt_b"]
    response_b = row["response_b"]

    tasks = [
        _eval_one_dimension(
            dim, prompt_a, response_a, prompt_b, response_b,
            client, semaphore, model, system_prompt, user_prompt_template,
            row_index,
        )
        for dim in dimensions
    ]
    results = await asyncio.gather(*tasks)

    scores = {}
    for dim, result in zip(dimensions, results):
        if result is not None:
            scores[dim["key"]] = result
        else:
            scores[dim["key"]] = {"score_a": -1, "score_b": -1, "error": True}

    return {
        "row_index":  row_index,
        "model":      row["model"],
        "prompt_a":   prompt_a,
        "response_a": response_a,
        "prompt_b":   prompt_b,
        "response_b": response_b,
        "scores":     scores,
    }


# ── Batch runner ───────────────────────────────────────────────────────────────

async def _eval_batch(
    rows: list,
    dimensions: list[dict],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    system_prompt: str,
    user_prompt_template: str,
    output_path: str,
    done_indices: set,
) -> None:
    remaining = [r for r in rows if r["row_index"] not in done_indices]
    total     = len(remaining)
    completed = 0
    t0        = time.time()

    n_dims = len(dimensions)
    print(
        f"[SCORE] Starting: {total} rows × {n_dims} dimensions = {total * n_dims:,} API calls  "
        f"model={model}  pool_size={semaphore._value}",
        flush=True,
    )

    batch_size = max(1, semaphore._value // 2)
    lock = asyncio.Lock()

    async def _process(row: dict) -> None:
        nonlocal completed
        result = await _eval_row(
            row, dimensions, client, semaphore, model, system_prompt, user_prompt_template
        )
        async with lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            completed += 1
            elapsed = time.time() - t0
            rate    = completed / elapsed
            eta_m   = (total - completed) / rate / 60 if rate > 0 else float("inf")

            diffs = {
                k: round(v["score_a"] - v["score_b"], 2)
                for k, v in result["scores"].items()
                if "error" not in v
            }
            top = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            top_str = "  ".join(f"{k}={d:+.2f}" for k, d in top)
            print(
                f"[SCORE] {completed}/{total}  row={row['row_index']}"
                f"  top_diffs: {top_str}"
                f"  {rate:.2f} row/s  ETA {eta_m:.1f}min",
                flush=True,
            )

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]
        await asyncio.gather(*[_process(r) for r in batch], return_exceptions=True)


# ── Response-only template helper ──────────────────────────────────────────────

_QUESTION_A_BLOCK = "Question_A:\n{question_a}\n\n"
_QUESTION_B_BLOCK = "Question_B:\n{question_b}\n\n"


def _make_response_only_template(template: str) -> str:
    """Remove Question_A/B blocks from the user prompt template."""
    t = template.replace(_QUESTION_A_BLOCK, "")
    t = t.replace(_QUESTION_B_BLOCK, "")
    return t


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score LLM responses with GRAPH rubric")
    parser.add_argument("--response_only", action="store_true",
                        help="Hide questions from judge; evaluate responses alone")
    args = parser.parse_args()

    input_path = os.environ.get(
        "INPUT",
        os.path.join(REPO_ROOT, "data", "response", "response_gpt-5.4-mini.csv"),
    )
    rubric_path = os.environ.get("RUBRIC", RUBRIC_PATH)
    limit       = int(os.environ.get("LIMIT", "1000"))
    pool_size   = int(os.environ.get("POOL_SIZE", "32"))

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    openai_key     = os.environ.get("OPENAI_API_KEY", "")

    if openrouter_key:
        api_key  = openrouter_key
        base_url = "https://openrouter.ai/api/v1"
        model    = os.environ.get("MODEL", "openai/gpt-4o-mini")
        print("[SCORE] Using OpenRouter API", flush=True)
    elif openai_key:
        api_key  = openai_key
        base_url = None
        model    = os.environ.get("MODEL", "gpt-4o-mini")
        print("[SCORE] Using OpenAI API", flush=True)
    else:
        raise RuntimeError("Set OPENROUTER_API_KEY or OPENAI_API_KEY.")

    system_prompt, user_prompt_template, dimensions = load_rubric(rubric_path)

    if args.response_only:
        user_prompt_template = _make_response_only_template(user_prompt_template)
        print("[SCORE] Response-only mode: Question_A/B hidden from judge", flush=True)

    dim_keys = [d["key"] for d in dimensions]
    print(f"[SCORE] Rubric: {rubric_path}  ({len(dimensions)} dimensions: {dim_keys})", flush=True)

    # Build output path
    base_name = os.path.splitext(os.path.basename(input_path))[0]  # e.g. "response_gpt-5.4-mini"
    suffix    = "_response_only" if args.response_only else ""
    subdir    = "response_only" if args.response_only else "full"
    output_dir  = os.path.join(RESULTS_DIR, "scores", subdir)
    output_path = os.path.join(output_dir, f"score_{base_name}{suffix}.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    # Load input CSV
    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = rows[:limit]
    for i, r in enumerate(rows):
        r["row_index"] = i

    print(f"[SCORE] Input : {input_path}  ({len(rows)} rows)", flush=True)
    print(f"[SCORE] Output: {output_path}", flush=True)

    # Resume: skip already-scored rows
    done_indices: set = set()
    if os.path.isfile(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_indices.add(json.loads(line)["row_index"])
                    except (KeyError, json.JSONDecodeError):
                        pass
        if done_indices:
            print(f"[SCORE] Resume: {len(done_indices)} rows already scored, skipping", flush=True)

    client    = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0)
    semaphore = asyncio.Semaphore(pool_size)

    asyncio.run(_eval_batch(
        rows=rows,
        dimensions=dimensions,
        client=client,
        semaphore=semaphore,
        model=model,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        output_path=output_path,
        done_indices=done_indices,
    ))

    print(f"\n[SCORE] Done: {output_path}", flush=True)


if __name__ == "__main__":
    main()
