#!/usr/bin/env bash

set -euo pipefail

model_name=PBT
topK=-1
finetune_dataset=ZN-coin42
batch_size=128
args_path=${ARGS_PATH:-/data/LLMs/checkpoints/Llama/pbt_mixL_lma_s42_r100/}
master_port=25251
train_epochs=300

seq_len=1
early_cycle_threshold=100
learning_rate=0.00005
warm_up_epoches=0
adapter_size=16
adapter_layers=-1
wd=0.0
dropout=0.0
loss=MSE
lradj_factor=0.5
finetune_method=FT
num_domains=32
aug_w=1.0
temperature=1.0
down_sample_ratio=0.75

llm_layers=32
num_process=1
accumulation_steps=1
bottleneck_factor=16

charge_discharge_length=300
patience=30
lradj=constant
top_p=0.5
gamma=1.0
patch_len=10
stride=10
least_epochs=50
embed=Cycle
P_token_num=4
activation=relu

num_views=4
num_hyper_experts=0
num_condition_experts=0
num_general_experts=5
num_experts=20
cathode_experts=11
temperature_experts=14
format_experts=11
anode_experts=11
ion_experts=0
cycle_topK=2
importance_weight=1.0
checkpoints=/data/tmpf
data=Dataset_BatteryLifeLLM_original
root_path=/data/trf/python_works/PBT_BatteryLife/dataset
comment='agent-b2'

# Agent
agent_cache_path=./agent/cache_mix_large_llama.pkl
agent_conf_threshold=0.25
agent_embed_mix=0.5
agent_gate_bias_scale=1.0
agent_use_curriculum=true
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

CUDA_VISIBLE_DEVICES=4 accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port finetune_model.py \
  --num_process $num_process \
  --lradj_factor $lradj_factor \
  --task_name battery_life_prediction \
  --data $data \
  --importance_weight $importance_weight \
  --is_training 1 \
  --root_path $root_path \
  --num_experts $num_experts \
  --topK $topK \
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
  --embed $embed \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --accumulation_steps $accumulation_steps \
  --charge_discharge_length $charge_discharge_length \
  --num_workers 16 \
  --num_views $num_views \
  --patience $patience \
  --early_cycle_threshold $early_cycle_threshold \
  --dropout $dropout \
  --lradj $lradj \
  --loss $loss \
  --checkpoints $checkpoints \
  --wd $wd \
  --least_epochs $least_epochs \
  --gamma $gamma \
  --warm_up_epoches $warm_up_epoches \
  --aug_w $aug_w \
  --down_sample_ratio $down_sample_ratio \
  --args_path $args_path \
  --finetune_dataset $finetune_dataset \
  --temperature $temperature \
  --finetune_method $finetune_method \
  --adapter_size $adapter_size \
  --adapter_layers $adapter_layers \
  --use_agent \
  --agent_cache_path $agent_cache_path \
  --agent_conf_threshold $agent_conf_threshold \
  --agent_embed_mix $agent_embed_mix \
  --agent_gate_bias_scale $agent_gate_bias_scale \
  "${extra_agent_flags[@]}"
