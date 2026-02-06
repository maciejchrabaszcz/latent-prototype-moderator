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
    mistral12b_inst
    olmo2_1124_13b_inst
)


layer_ids=({1..40})
seeds=({0..4})
source .venv/bin/activate

# Run for all seeds
for seed_id in ${seeds[@]}; do
    for model in "${models[@]}"; do
        for layer_id in "${layer_ids[@]}"; do
            echo "Running prototype based preds for model: $model, layer: $layer_id", seed: $seed_id
            # check if the layer file exists
            layer_file=$MY_DATA/datasets/wildguard_hidden_states/$model/layer_${layer_id}_hidden_states.parquet
            if [ ! -f "$layer_file" ]; then
                echo "Layer file $layer_file does not exist. Skipping."
                continue
            fi
            python scripts/hidden_states/eval_prototype_based_classificaiton.py \
                --train_hidden_states_folder $MY_DATA/datasets/wildguard_hidden_states/$model \
                --save_folder $MY_DATA/results/prototypes/per_layer_seeds/$model/seed_$seed_id/layer_$layer_id \
                --harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/harmful/ \
                --non_harmful_benchmarks_folder $MY_DATA/datasets/eval_datasets_hidden_states/$model/general_benchmarks/ \
                --layer_file_name layer_${layer_id}_hidden_states.parquet \
                --num_frac_data 0.9

        done
    done
done