#!/bin/bash

models=(
    # meta-llama/Llama-Guard-3-8B
    # allenai/wildguard
    nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0
)


save_folders=(
    # llama_guard_3_8b
    # wildguard
    aegis_defensive
)


source .venv/bin/activate

for i in "${!models[@]}"; do
    save_folder=${save_folders[$i]}
    model=${models[$i]}

    echo "Running evaluation for model: $model, save_folder: $save_folder"

    python scripts/evaluation/generate_llm_classifiers_outputs.py \
        --base_model $model \
        --save_folder $MY_DATA/results/baseline_scores/$save_folder \
        --harmful_benchmarks_folder data/processed_benchmarks/harmfulness/prompt/ \
        --non_harmful_benchmarks_folder data/processed_benchmarks/general_capabilities/ \
        --batch_size 2
done
