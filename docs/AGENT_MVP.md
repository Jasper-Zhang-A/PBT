# Agent MVP Runbook

This runbook covers the offline Battery Condition Compiler Agent workflow for PBT.

The first cache is built entirely from assets already present in this repo and dataset layout:

- existing DKP embedding pickles
- `gate_data/*.json`
- `data_provider/gate_masker.py`
- `data_provider/data_split_recorder.py`

No extra LLM downloads are required for cache construction.

## What The Agent Adds

The following flags are available in `run_main.py`, `finetune_model.py`, and `evaluate_model.py`:

- `--use_agent`: enable offline agent features
- `--agent_cache_path`: pickle path for the offline agent cache
- `--agent_conf_threshold`: confidence threshold for applying gate-prior bias
- `--agent_embed_mix`: max blend ratio between DKP embedding and agent condition embedding
- `--agent_gate_bias_scale`: scale factor applied to normalized gate prior before routing
- `--agent_use_curriculum`: multiplies existing sample weights by `agent_sample_weight`
- `--agent_use_gate_prior`: enables routing bias from `agent_gate_prior`
- `--agent_debug`: prints one debug line per process on the first batch

## Cache Schema

The cache is a pickle dictionary with both key forms supported:

```python
{
    "FILE_NAME.pkl": {
        "cond_embed": [...],          # length d_llm
        "gate_prior": [...],          # length 65 in the current PBT mask layout
        "confidence": 0.0,            # float in [0, 1]
        "curriculum_weight": 1.0,     # float
    },
    "CELL_NAME_WITHOUT_PKL": {
        ...
    },
}
```

Fallbacks used by `agent/cache.py`:

- `cond_embed`: existing DKP embedding
- `gate_prior`: normalized `combined_mask` if its sum is positive, else all zeros
- `confidence`: `0.0`
- `curriculum_weight`: `1.0`

In this repo, `combined_mask` is the concatenation of:

- cathode mask with length `13`
- anode mask with length `11`
- format mask with length `21`
- temperature mask with length `20`

So the current `gate_prior` length is `65`.

## Step 1: Build Cache

Direct command:

```bash
python agent/build_agent_cache.py \
  --root_path /data/trf/python_works/BatteryLife/dataset \
  --llm_choice Llama \
  --mode mix_large \
  --dataset MIX_large \
  --rarity_alpha 1.0 \
  --output_path ./agent/cache_mix_large_llama.pkl
```

Scripted command:

```bash
bash scripts/build_agent_cache.sh
```

The builder is fully offline and reuses only:

- `training_DKP_embed_all_{llm_choice}.pkl`
- `validation_DKP_embed_all_{llm_choice}.pkl`
- `testing_DKP_embed_all_{llm_choice}.pkl`
- `gate_data/*.json`
- `data_provider/gate_masker.py`
- `data_provider/data_split_recorder.py`

Builder behavior:

- `cond_embed` is copied from existing DKP embeddings
- `gate_prior` is a normalized cathode/anode/format/temperature prior
- `confidence` is `0.25` per successfully mapped field
- `curriculum_weight` is `1 + rarity_alpha / sqrt(freq)`
- `freq` comes from `gate_data/name2agingConditionID.json`

## Step 2A: Pretrain / Train From Scratch

PBT:

```bash
bash scripts/PBT_agent.sh
```

CPMLP:

```bash
bash scripts/CPMLP_agent.sh
```

CPTransformer:

```bash
bash scripts/CPTransformer_agent.sh
```

These scripts mirror the existing repo scripts and add the agent flags on top.

## Step 2B: Finetune

PBT finetuning:

```bash
bash scripts/finetune_PBT_agent.sh
```

If you prefer the direct command style, set:

- `args_path` to the pretrained checkpoint directory
- `finetune_dataset` to the target dataset
- `agent_cache_path` to the cache built in step 1

The shipped finetune script already includes those variables at the top for editing.

## Step 3: Evaluate

Evaluation for both baseline models and PBT:

```bash
bash scripts/evaluate_agent.sh
```

This mirrors the existing `scripts/evaluate_model.sh` pattern and adds agent flags.
It works for:

- PBT runs
- CPMLP runs
- CPTransformer runs

because `evaluate_model.py` now forwards the agent inputs to all three model families.

## Script Map

- `scripts/build_agent_cache.sh`
  offline cache construction
- `scripts/PBT_agent.sh`
  PBT training with agent flags
- `scripts/finetune_PBT_agent.sh`
  PBT finetuning with agent flags
- `scripts/CPMLP_agent.sh`
  CPMLP training with agent flags
- `scripts/CPTransformer_agent.sh`
  CPTransformer training with agent flags
- `scripts/evaluate_agent.sh`
  evaluation with agent flags

## Changed Code Paths

- `data_provider/data_loader.py`
  loads the cache once and appends agent fields to samples and batches
- `models/CPMLP.py`
  late-fuses projected agent embedding after `inter_MLP`
- `models/CPTransformer.py`
  late-fuses projected agent embedding after flattened inter-cycle representation
- `models/PBT.py`
  mixes DKP with agent embedding and applies optional routing bias through MoE layers
- `run_main.py`
  training loop forwards agent tensors and optional curriculum weights
- `finetune_model.py`
  finetuning loop forwards agent tensors and preserves agent flags after loading saved args
- `evaluate_model.py`
  evaluation loop forwards agent tensors and preserves agent flags after loading saved args
- `utils/tools.py`
  validation helpers forward agent tensors and handle agent debug printing
