from datasets import load_dataset


def load_wildguard_for_prompt_classification():
    dataset = load_dataset(
        "allenai/wildguardmix", "wildguardtrain", keep_in_memory=True
    )["train"]

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
        example["labels"] = float(example["prompt_harm_label"] == "harmful")
        example["category"] = example["subcategory"]
        return example

    dataset = dataset.map(
        map_dataset,
        remove_columns=dataset.column_names,
    )
    return dataset


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
        example["category"] = example["subcategory"]
        return example

    dataset = dataset.map(
        map_dataset,
        remove_columns=dataset.column_names,
    )
    return dataset

def load_wildguardtest_prompt(include_response=True) -> list[dict]:
    df = load_dataset("allenai/wildguardmix", "wildguardtest")["test"]
    df = df.to_pandas()
    # drop if prompt_harm_label is None
    df = df.dropna(subset=["prompt_harm_label"])
    # setup prompt_type based on "adversarial"
    df["prompt_type"] = df.apply(
        lambda row: "adversarial" if row["adversarial"] else "vanilla", axis=1
    )
    df["gt_label"] = df["prompt_harm_label"]

    return df




def load_wildguardtest_response():
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

    return df