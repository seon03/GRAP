#!/usr/bin/env bash
# run.sh — End-to-end GRAP pipeline
#
# Steps:
#   1. Generate LLM responses        (src/generate_responses.py)
#   2. Score responses with rubric   (src/score.py)
#   3. Statistical analysis + paper figures (src/analyze.py)
#
# Pre-generated response and score data are already in data/response/ and
# results/scores/, so you can skip steps 1–2 and run only step 3.
#
# Environment variables are loaded from .env (copy .env.example → .env first):
#   OPENROUTER_API_KEY   for generating Gemini/Claude/GPT responses and scoring
#   VLLM_PORT            for generating LLaMA responses (default: 8000)
#
# Usage examples:
#   # Full pipeline from scratch
#   bash run.sh
#
#   # Analysis + figures only (skip response generation and scoring)
#   bash run.sh --skip-generate --skip-score

set -e
cd "$(dirname "$0")"

# ── Load .env ──────────────────────────────────────────────────────────────────
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "[WARN] .env not found. Copy .env.example to .env and fill in your API keys."
fi

SKIP_GENERATE=false
SKIP_SCORE=false

for arg in "$@"; do
  case "$arg" in
    --skip-generate) SKIP_GENERATE=true ;;
    --skip-score)    SKIP_SCORE=true ;;
  esac
done

# ── Step 1: Generate responses ─────────────────────────────────────────────────
if [ "$SKIP_GENERATE" = false ]; then
  echo "=== Step 1: Generate responses ==="

  echo "  Generating Gemini 2.5 Flash responses..."
  python src/generate_responses.py --model gemini --pool 16

  echo "  Generating Claude Sonnet 4.5 responses..."
  python src/generate_responses.py --model claude --pool 16

  echo "  Generating GPT-5.4-mini responses..."
  python src/generate_responses.py --model gpt --pool 16

  echo "  Generating LLaMA 3.3 70B responses (requires local vllm server)..."
  echo "  Start the server first:"
  echo "    vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 4 --port \${VLLM_PORT:-8000}"
  python src/generate_responses.py --model llama --pool 32
else
  echo "=== Step 1: Skipped (using pre-generated responses in data/response/) ==="
fi

# ── Step 2: Score responses ────────────────────────────────────────────────────
if [ "$SKIP_SCORE" = false ]; then
  echo ""
  echo "=== Step 2: Score responses with rubric (GPT-4o-mini judge) ==="

  for MODEL in gpt-5.4-mini claude-sonnet-4-5 gemini-2.5-flash llama-3.3-70b-instruct; do
    echo "  Scoring ${MODEL} (full evaluation)..."
    INPUT="data/response/response_${MODEL}.csv" \
      POOL_SIZE=32 \
      python src/score.py

    echo "  Scoring ${MODEL} (response-only evaluation)..."
    INPUT="data/response/response_${MODEL}.csv" \
      POOL_SIZE=32 \
      python src/score.py --response_only
  done
else
  echo "=== Step 2: Skipped (using pre-computed scores in results/scores/) ==="
fi

# ── Step 3: Statistical analysis + paper figures ───────────────────────────────
echo ""
echo "=== Step 3: Statistical analysis + paper figures (full evaluation) ==="
python src/analyze.py \
  --score_dir results/scores/full \
  --out_dir   results/analysis \
  --remove_outliers 200

echo ""
echo "=== Step 3b: Statistical analysis (response-only evaluation) ==="
python src/analyze.py \
  --score_dir results/scores/response_only \
  --out_dir   results/analysis \
  --remove_outliers 200

echo ""
echo "=== Pipeline complete ==="
echo "Figures saved to results/analysis/"
