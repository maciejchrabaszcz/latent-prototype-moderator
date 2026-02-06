#!/bin/bash

models=(
    # meta-llama/Llama-3.1-8B
    # meta-llama/Llama-3.1-8B-Instruct
    # mistralai/Mistral-7B-v0.3
    # mistralai/Mistral-7B-Instruct-v0.3
    # mistralai/Mistral-Nemo-Instruct-2407
    # allenai/OLMo-2-1124-7B
    # allenai/OLMo-2-1124-7B-SFT
    # allenai/OLMo-2-1124-7B-DPO
    # allenai/OLMo-2-1124-7B-Instruct
    # allenai/OLMo-2-1124-13B-Instruct
    # allenai/OLMoE-1B-7B-0125-Instruct
    allenai/OLMo-2-0325-32B-Instruct
    # allenai/OLMo-2-0425-1B-Instruct
    # meta-llama/Llama-Guard-3-8B
    # allenai/wildguard
)

save_folders=(
    # llama3_8b
    # llama3_8b_inst
    # mistral7b
    # mistral7b_inst
    # mistral12b_inst
    # olmo2_1124_7b
    # olmo2_1124_7b_sft
    # olmo2_1124_7b_dpo
    # olmo2_1124_7b_inst
    # olmo2_1124_13b_inst
    # olmoe_1b_7b_0125_inst
    olmo2_0325_32b_inst
    # olmo2_0425_1b_inst
    # llama_guard_3_8b
    # wildguard
)
batch_sizes=(
    # 4
    # 4
    # 4
    # 4
    # 4
    # 4
    # 4
    # 4
    # 4
    # 4
    # 16
    1
    # 16
    # 4
    # 4
)


source .venv/bin/activate

for i in "${!models[@]}"; do
    save_folder=${save_folders[$i]}
    model=${models[$i]}
    batch_size=${batch_sizes[$i]}
    echo "Running generation for model: $model, save_folder: $save_folder"

    python scripts/hidden_states/generate_hidden_states_for_train.py \
        --base_model $model \
        --dataset_path data/wildguardmix_response \
        --save_folder $MY_DATA/datasets/wildguard_responses_hidden_states/$save_folder \
        --response_dataset \
        --batch_size $batch_size

done

# bash bash_scripts/generate_models_hidden_states_responses.sh && bash bash_scripts/generate_models_hidden_states_responses_eval_data.sh && bash bash_scripts/generate_models_hidden_states_eval_data.sh && bash bash_scripts/generate_train_dataset_hidden_states.sh
