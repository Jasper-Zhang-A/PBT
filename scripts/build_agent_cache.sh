#!/usr/bin/env bash

set -euo pipefail

ROOT_PATH=/data/trf/python_works/BatteryLife/dataset
LLM_CHOICE=Llama
MODE=mix_large
DATASET=MIX_large
RARITY_ALPHA=1.0
OUTPUT_PATH=./agent/cache_mix_large_llama.pkl

python agent/build_agent_cache.py \
  --root_path "$ROOT_PATH" \
  --llm_choice "$LLM_CHOICE" \
  --mode "$MODE" \
  --dataset "$DATASET" \
  --rarity_alpha "$RARITY_ALPHA" \
  --output_path "$OUTPUT_PATH"
