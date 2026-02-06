import argparse
from activation_steering import SteeringDataset, SteeringVector
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Steering Vector")
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the behavior vector",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model ID for the transformer model",
    )
    return parser.parse_args()


# %%
num_training_samples = 5000


def load_wildguardtest():
    df = load_dataset("allenai/wildguardmix", "wildguardtest")["test"]
    df = df.to_pandas()
    # drop if prompt_harm_label is None
    df = df.dropna(subset=["response_harm_label"])
    # setup prompt_type based on "adversarial"
    df["prompt_type"] = df.apply(
        lambda row: "adversarial" if row["adversarial"] else "vanilla", axis=1
    )
    df["gt_label"] = df["response_harm_label"]
    df["prompt_harm_category"] = df["subcategory"]

    return [
        {
            "prompt": row["prompt"],
            "response": row["response"],
            "response_harmfulness": True if row["gt_label"] == "harmful" else False,
            "prompt_type": row["prompt_type"],
            "prompt_harmfulness": row["prompt_harm_label"] == "harmful",
            "prompt_harm_category": row["subcategory"],
        }
        for i, row in df.iterrows()
    ]


def load_wildguard_for_response_classification():
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
            {
                "role": "assistant",
                "content": example["response"],
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

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", dtype="auto", trust_remote_code=True
    )

    # %%
    train_ds = load_wildguard_for_response_classification()
    train_ds = train_ds.filter(lambda example: example["prompt_label"] == 1.0)

    # %%
    # get num_training_samples for each labels
    shuffled_ds = train_ds.shuffle(seed=42)
    negative_samples = shuffled_ds.filter(lambda example: example["labels"] == 0).select(
        range(num_training_samples)
    )["messages"]
    positive_samples = shuffled_ds.filter(lambda example: example["labels"] == 1).select(
        range(num_training_samples)
    )["messages"]

    # %%
    behavior_dataset = SteeringDataset(
        tokenizer=tokenizer,
        examples=[(x, y) for x, y in zip(negative_samples, positive_samples)],
        suffixes=[
            (x[1]["content"], y[1]["content"])
            for x, y in zip(negative_samples, positive_samples)
        ],
        disable_suffixes=True,
        already_pairs=True,
    )

    # %%
    behavior_vector = SteeringVector.train(
        model=model,
        tokenizer=tokenizer,
        steering_dataset=behavior_dataset,
        method="pca_pairwise",
        accumulate_last_x_tokens="suffix-only",
        batch_size=4,
    )

    # %%
    behavior_vector.save(save_path)
