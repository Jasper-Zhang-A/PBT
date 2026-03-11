#!/usr/bin/env bash

set -euo pipefail

model_name=CPMLP
master_port=25291
dataset=ZN-coin2024
seed=2024
train_epochs=100
warm_up_epoches=0
early_cycle_threshold=100
learning_rate=0.00005
lradj_factor=0.5

num_domains=32
aug_w=1.0
temperature=1.0
down_sample_ratio=0.75

llm_layers=32
tune_layers=4

num_process=2
batch_size=128
n_heads=8
seq_len=1
accumulation_steps=1

e_layers=4
d_layers=6

bottleneck_factor=16
d_model=128
d_ff=64

d_llm=4096
dropout=0.25
charge_discharge_length=300
patience=5
lradj=constant
top_p=0.5
gamma=1.0
loss=MSE
patch_len=10
stride=10
wd=0.0
least_epochs=10
embed=Cycle
P_token_num=4
activation=relu

LLM_path=/data/LLMs/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
checkpoints=/data/LLMs/checkpoints
data=Dataset_PBT
root_path=/data/trf/python_works/BatteryLife/dataset
comment='agent-50to1'

# Agent
agent_cache_path=./agent/cache_mix_large_llama.pkl
agent_conf_threshold=0.25
agent_embed_mix=0.5
agent_gate_bias_scale=1.0
agent_use_curriculum=true
agent_use_gate_prior=false
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

CUDA_VISIBLE_DEVICES=4,5 accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --num_process $num_process \
  --lradj_factor $lradj_factor \
  --task_name battery_life_prediction \
  --data $data \
  --is_training 1 \
  --root_path $root_path \
  --model_id TunePara \
  --model $model_name \
  --features MS \
  --seq_len $seq_len \
  --label_len 50 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --seed $seed \
  --embed $embed \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llm_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --accumulation_steps $accumulation_steps \
  --charge_discharge_length $charge_discharge_length \
  --dataset $dataset \
  --num_workers 16 \
  --d_llm $d_llm \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --patience $patience \
  --n_heads $n_heads \
  --early_cycle_threshold $early_cycle_threshold \
  --dropout $dropout \
  --lradj $lradj \
  --loss $loss \
  --checkpoints $checkpoints \
  --LLM_path $LLM_path \
  --patch_len $patch_len \
  --stride $stride \
  --wd $wd \
  --least_epochs $least_epochs \
  --activation $activation \
  --top_p $top_p \
  --gamma $gamma \
  --warm_up_epoches $warm_up_epoches \
  --use_agent \
  --agent_cache_path $agent_cache_path \
  --agent_conf_threshold $agent_conf_threshold \
  --agent_embed_mix $agent_embed_mix \
  --agent_gate_bias_scale $agent_gate_bias_scale \
  "${extra_agent_flags[@]}"
