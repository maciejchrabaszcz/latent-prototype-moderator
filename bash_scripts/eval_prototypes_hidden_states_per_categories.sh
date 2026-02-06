#!/bin/bash


models=(
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

bigger_models=(
    mistral12b_inst
    olmo2_1124_13b_inst
)

source .venv/bin/activate

for i in "${!models[@]}"; do
    model=${models[$i]}

    echo "Running prototype based preds for model: $model"

    python scripts/hidden_states/eval_prototype_based_with_categories.py \
        --train_hidden_states_folder $MY_DATA/datasets/wildguard_hidden_states/$model \
        --save_folder $MY_DATA/results/prototypes_per_ordered_categories/$model \
        --harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/harmful/ \
        --non_harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/general_benchmarks/ \
        --layer_file_name layer_32_hidden_states.parquet

done

for i in "${!bigger_models[@]}"; do
    model=${bigger_models[$i]}

    echo "Running prototype based preds for model: $model"

    python scripts/hidden_states/eval_prototype_based_with_categories.py \
        --train_hidden_states_folder $MY_DATA/datasets/wildguard_hidden_states/$model \
        --save_folder $MY_DATA/results/prototypes_per_ordered_categories/$model \
        --harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/harmful/ \
        --non_harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/general_benchmarks/ \
        --layer_file_name layer_40_hidden_states.parquet

done