#!/bin/bash

source ../.steer_venv/bin/activate

models=(
    mistralai/Mistral-7B-Instruct-v0.3
    meta-llama/Llama-3.1-8B-Instruct
    Qwen/Qwen3-8B
)

save_names=(
    mistral_7b_inst
    llama3_8b_inst
    qwen3_8b
)

for i in "${!models[@]}"; do
    model=${models[$i]}
    save_name=${save_names[$i]}
    echo "Calculating condition vectors for model: $model, save_name: $save_name"
    python calculate_condition_vector.py \
        --model_id "$model" \
        --save_path conditioning/$save_name
done