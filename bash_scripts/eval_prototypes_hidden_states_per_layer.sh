#!/bin/bash


models=(
    # llama3_8b
    # llama3_8b_inst
    # mistral7b
    # mistral7b_inst
    # olmo2_1124_7b
    # olmo2_1124_7b_sft
    # olmo2_1124_7b_dpo
    # olmo2_1124_7b_inst
    # olmo2_1124_13b_inst
    # mistral12b_inst
    # olmoe_1b_7b_0125_inst
    olmo2_0325_32b_inst
    # olmo2_0425_1b_inst
)

layer_ids=({1..60})

source .venv/bin/activate

for model in "${models[@]}"; do
    for layer_id in "${layer_ids[@]}"; do
        echo "Running prototype based preds for model: $model, layer: $layer_id"

        python scripts/hidden_states/eval_prototype_based_classificaiton.py \
            --train_hidden_states_folder $MY_DATA/datasets/wildguard_hidden_states/$model \
            --save_folder $MY_DATA/results/per_layer_prototypes/$model/layer_$layer_id \
            --harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/harmful/ \
            --non_harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/general_benchmarks/ \
            --layer_file_name layer_${layer_id}_hidden_states.parquet
    done
done
