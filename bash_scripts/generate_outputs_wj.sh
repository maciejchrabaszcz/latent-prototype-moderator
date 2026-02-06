#!/bin/bash

models=(
    # meta-llama/Llama-3.1-8B
    meta-llama/Llama-3.1-8B-Instruct
    # mistralai/Mistral-7B-v0.3
    mistralai/Mistral-7B-Instruct-v0.3
    # allenai/OLMo-2-1124-7B
    # allenai/OLMo-2-1124-7B-SFT
    # allenai/OLMo-2-1124-7B-DPO
    allenai/OLMo-2-1124-7B-Instruct
)

save_folders=(
    # llama3_8b
    llama3_8b_inst
    # mistral7b
    mistral7b_inst
    # olmo2_1124_7b
    # olmo2_1124_7b_sft
    # olmo2_1124_7b_dpo
    olmo2_1124_7b_inst
    # llama_guard_3_8b
    # wildguard
)


source .venv/bin/activate

for i in "${!models[@]}"; do
    save_folder=${save_folders[$i]}
    model=${models[$i]}

    echo "Running generation for model: $model"

    python scripts/generation/generate_wj_resposnes.py \
        --base_model $model \
        --output_file $MY_DATA/results/wj_generations/"$save_folder"_preds.json \
        --add_generation_prompt \
        --batch_size 2500
done
