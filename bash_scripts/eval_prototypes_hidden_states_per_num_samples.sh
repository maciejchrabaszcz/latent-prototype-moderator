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
)
seeds_idx=({0..4})
num_samples_values=(
    # 2
    # 3
    # 4
    # 8
    10
    50
    100
    500
    1000
    5000
    10000
    20000
    40000
)

source .venv/bin/activate

for seed_id in ${seeds_idx[@]}; do
    for model in "${models[@]}"; do
        for num_samples in "${num_samples_values[@]}"; do
            echo "Running prototype based preds for model: $model, num samples: $num_samples", seed: $seed_id

            python scripts/hidden_states/eval_prototype_based_classificaiton.py \
                --train_hidden_states_folder $MY_DATA/datasets/wildguard_hidden_states/$model \
                --save_folder $MY_DATA/results/prototypes_different_num_samples_seeds/$model/seed_$seed_id/"$num_samples"_samples \
                --harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/harmful/ \
                --non_harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/general_benchmarks/ \
                --layer_file_name layer_32_hidden_states.parquet \
                --num_samples_per_class $num_samples
        done
    done
done
