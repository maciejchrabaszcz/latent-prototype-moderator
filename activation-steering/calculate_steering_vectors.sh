#!/bin/bash

source ../.steer_venv/bin/activate

models=(

    meta-llama/Llama-3.1-8B-Instruct
    Qwen/Qwen3-8B
)

save_names=(
    llama3_8b_inst
    qwen3_8b
)

for i in "${!models[@]}"; do
    model=${models[$i]}
    save_name=${save_names[$i]}
    echo "Calculating steering vectors for model: $model, save_name: $save_name"
    python calculate_steering_vector.py \
        --model_id "$model" \
        --save_path behavior_vector_$save_name
done