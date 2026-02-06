#!/bin/bash


models=(
    # llama3_8b
    # llama3_8b_inst
    # mistral7b
    # mistral7b_inst
    mistral12b_inst
    # olmo2_1124_7b
    # olmo2_1124_7b_sft
    # olmo2_1124_7b_dpo
    # olmo2_1124_7b_inst
    olmo2_1124_13b_inst
    # llama_guard_3_8b
    # wildguard
)



source .venv/bin/activate

for i in "${!models[@]}"; do
    model=${models[$i]}

    echo "Running prototype based preds for model: $model"

    python scripts/hidden_states/eval_prototype_based_classificaiton.py \
        --train_hidden_states_folder $MY_DATA/datasets/wildguard_responses_hidden_states/$model \
        --save_folder $MY_DATA/results/response_prototypes/$model \
        --harmful_benchmarks_folder $MY_DATA/datasets/responses_eval_datasets_hidden_states/$model \
        --layer_file_name layer_40_hidden_states.parquet

done
