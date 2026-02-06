#!/bin/bash


models=(
    # llama3_8b
    # llama3_8b_inst
    # mistral7b
    # mistral7b_inst
    # mistral12b_inst
    # olmo2_1124_7b
    # olmo2_1124_7b_sft
    # olmo2_1124_7b_dpo
    # llama_guard_3_8b
    # wildguard
    # olmo2_1124_7b_inst
    # olmo2_1124_13b_inst
    # olmoe_1b_7b_0125_inst
    # olmo2_0325_32b_inst
    # olmo2_0425_1b_inst
    llama3_70b_inst
    # llama3_3b_inst
    # llama3_1b_inst
)

layer_to_use=(
    # 32
    # 32
    # 32
    # 32
    # 40
    # 32
    # 25
    # 32
    # 32
    # 40
    # 16
    # 50
    # 16
    80
    # 28
    # 16
    # 16
)
source .venv/bin/activate

for i in "${!models[@]}"; do
    model=${models[$i]}
    layer=${layer_to_use[$i]}
    echo "Running prototype based preds for model: $model"
    echo layer_"$layer"_hidden_states.parquet
    python scripts/hidden_states/eval_prototype_based_classificaiton.py \
        --train_hidden_states_folder $MY_DATA/datasets/wildguard_hidden_states/$model \
        --save_folder $MY_DATA/results/prototypes/$model \
        --harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/harmful/ \
        --non_harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/general_benchmarks/ \
        --layer_file_name layer_"$layer"_hidden_states.parquet

done
