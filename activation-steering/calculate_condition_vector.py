import argparse
import json
from pathlib import Path

from activation_steering import MalleableModel, SteeringDataset, SteeringVector
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


# %%
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Steering Vector")
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="Path to save the behavior vector",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model ID for the transformer model",
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=Path,
        default="../data/processed_benchmarks/harmfulness/prompt/wildguardtest",
        help="Path to the evaluation dataset",
    )
    return parser.parse_args()


# %%
num_training_samples = 5000


def load_wildguard_for_prompt_classification():
    dataset = load_dataset(
        "allenai/wildguardmix", "wildguardtrain", keep_in_memory=True
    )["train"]
    dataset = dataset.filter(
        lambda example: (example["response_harm_label"] is not None)
        and (example["response"] is not None)
    )

    def map_dataset(example):
        example["messages"] = [
            {
                "role": "user",
                "content": example["prompt"],
            },
        ]
        example["labels"] = float(example["response_harm_label"] == "harmful")
        example["prompt_label"] = float(example["prompt_harm_label"] == "harmful")
        example["category"] = example["subcategory"]
        return example

    dataset = dataset.map(
        map_dataset,
        remove_columns=dataset.column_names,
    )
    return dataset


# %%
if __name__ == "__main__":
    args = parse_args()
    model_id = args.model_id
    save_path = args.save_path
    eval_dataset_path = args.eval_dataset_path
    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", dtype="auto", trust_remote_code=True
    )

    # %%
    train_ds = load_wildguard_for_prompt_classification()

    # %%
    # get num_training_samples for each labels
    shuffled_ds = train_ds.shuffle(seed=42)
    negative_samples = shuffled_ds.filter(
        lambda example: example["prompt_label"] == 0
    ).select(range(num_training_samples))["messages"]
    positive_samples = shuffled_ds.filter(
        lambda example: example["prompt_label"] == 1
    ).select(range(num_training_samples))["messages"]

    # %%
    behavior_dataset = SteeringDataset(
        tokenizer=tokenizer,
        examples=[(x, y) for x, y in zip(positive_samples, negative_samples)],
        suffixes=None,
        disable_suffixes=True,
        already_pairs=True,
    )

    # %%
    condition_vector = SteeringVector.train(
        model=model,
        tokenizer=tokenizer,
        steering_dataset=behavior_dataset,
        method="pca_pairwise",
        accumulate_last_x_tokens="all",
        batch_size=4,
    )

    # %%
    condition_vector.save(str(save_path / "harmful_condition_vector"))

    malleable_model = MalleableModel(model=model, tokenizer=tokenizer)

    eval_ds = load_from_disk(eval_dataset_path)

    positive_eval_ds = eval_ds.filter(lambda example: example["prompt_harmfulness"])
    negative_eval_ds = eval_ds.filter(lambda example: not example["prompt_harmfulness"])
    positive_strings = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True
        )
        for x in positive_eval_ds["prompt"]
    ]
    negative_strings = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True
        )
        for x in negative_eval_ds["prompt"]
    ]
    best_layer, best_threshold, best_direction, _, analysis_results = (
        malleable_model.find_best_condition_point(
            positive_strings=positive_strings,
            negative_strings=negative_strings,
            condition_vector=condition_vector,
            layer_range=(4, 14),
            max_layers_to_combine=1,
            threshold_range=(0.0, 0.4),
            threshold_step=0.01,
            save_analysis=True,
            file_path=str(
                save_path / "optimal_condition_point_harmful_condition_vector.json"
            ),
            return_analysis=True,
        )
    )
    negative_preds = analysis_results["negative"]
    positive_preds = analysis_results["positive"]
    # Save preds in same order as eval_ds
    all_preds = []
    neg_idx = 0
    pos_idx = 0
    for example in eval_ds:
        if not example["prompt_harmfulness"]:
            all_preds.append(
                int(
                    negative_preds[neg_idx] > best_threshold
                    if best_direction == "larger"
                    else negative_preds[neg_idx] < best_threshold
                )
            )
            neg_idx += 1
        else:
            all_preds.append(
                int(
                    positive_preds[pos_idx] > best_threshold
                    if best_direction == "larger"
                    else positive_preds[pos_idx] < best_threshold
                )
            )
            pos_idx += 1
    with open(save_path / "wildguardtest_harmfulness_preds.json", "w") as f:
        json.dump(all_preds, f, indent=4, ensure_ascii=False)
