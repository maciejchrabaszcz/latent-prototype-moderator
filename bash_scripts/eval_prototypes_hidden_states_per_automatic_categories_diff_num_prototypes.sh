#!/bin/bash


models=(
    # llama3_8b
    # llama3_8b_inst
    # mistral7b
    # mistral7b_inst
    # olmo2_1124_7b
    # olmo2_1124_7b_sft
    # olmo2_1124_7b_dpo
    olmo2_1124_7b_inst
    # llama_guard_3_8b
    # wildguard
)

bigger_models=(
    # mistral12b_inst
    olmo2_1124_13b_inst
)
data_fractions=(
    0.03125
    0.0625
    0.125
    0.25
    0.5
    1.0
)
max_clusters=20

source .venv/bin/activate

for data_fraction in "${data_fractions[@]}"; do
    for model in "${models[@]}"; do

        echo "Running prototype based preds for model: $model", "data frac: $data_fraction"
        python scripts/hidden_states/eval_prototype_based_with_automatic_categories.py \
            --train_hidden_states_folder $MY_DATA/datasets/wildguard_hidden_states/$model \
            --save_folder $MY_DATA/results/prototypes_per_automatic_categories_ablation/$model/frac_$data_fraction \
            --harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/harmful/ \
            --non_harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/general_benchmarks/ \
            --layer_file_name layer_32_hidden_states.parquet \
            --data_fraction $data_fraction \
            --max_num_clusters $max_clusters
    done
done

for data_fraction in "${data_fractions[@]}"; do
    for model in "${bigger_models[@]}"; do

        echo "Running prototype based preds for model: $model", "data frac: $data_fraction"
        python scripts/hidden_states/eval_prototype_based_with_automatic_categories.py \
            --train_hidden_states_folder $MY_DATA/datasets/wildguard_hidden_states/$model \
            --save_folder $MY_DATA/results/prototypes_per_automatic_categories_ablation/$model/frac_$data_fraction \
            --harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/harmful/ \
            --non_harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/general_benchmarks/ \
            --layer_file_name layer_32_hidden_states.parquet \
            --data_fraction $data_fraction \
            --max_num_clusters $max_clusters
    done
done
