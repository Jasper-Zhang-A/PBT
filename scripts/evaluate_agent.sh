#!/usr/bin/env bash

set -euo pipefail

args_path=/data/LLMs/checkpoints/PBT_10_Llama_1_as16_al12_le50_bs16_lr5e-05_dm128_nh8_el2_dl10_df128_mdf64_lradjconstant_CALB2024_guideFalse_LBFalse_lossMSE_wd0.0_wlFalse_dr0.0_gdff512_E5_GE5_K-1_SFalse_augFalse_dsr0.75_ffsTrue_MIX_large_FT_seed2024-b2/
batch_size=128
num_process=1
master_port=24921
eval_cycle_min=20
eval_cycle_max=20
eval_dataset=CALB2024
root_path=/data/trf/python_works/PBT_BatteryLife/dataset

LLM_path=/data/LLMs/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659

# Agent
agent_cache_path=./agent/cache_mix_large_llama.pkl
agent_conf_threshold=0.25
agent_embed_mix=0.5
agent_gate_bias_scale=1.0
agent_use_curriculum=false
agent_use_gate_prior=true
agent_debug=false

extra_agent_flags=()
if [ "$agent_use_curriculum" = true ]; then
  extra_agent_flags+=(--agent_use_curriculum)
fi
if [ "$agent_use_gate_prior" = true ]; then
  extra_agent_flags+=(--agent_use_gate_prior)
fi
if [ "$agent_debug" = true ]; then
  extra_agent_flags+=(--agent_debug)
fi

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port evaluate_model.py \
  --args_path $args_path \
  --batch_size $batch_size \
  --eval_cycle_min $eval_cycle_min \
  --eval_cycle_max $eval_cycle_max \
  --eval_dataset $eval_dataset \
  --LLM_path $LLM_path \
  --root_path $root_path \
  --use_agent \
  --agent_cache_path $agent_cache_path \
  --agent_conf_threshold $agent_conf_threshold \
  --agent_embed_mix $agent_embed_mix \
  --agent_gate_bias_scale $agent_gate_bias_scale \
  "${extra_agent_flags[@]}"
