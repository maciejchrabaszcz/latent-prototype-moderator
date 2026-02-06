from pathlib import Path

import pandas as pd
from datasets import load_dataset

SAVE_FOLDER = Path("data/training_datasets")


def convert_aegis(example):
    label_0 = int(example["labels_0"] == "Safe")
    label_1 = int(example["labels_1"] == "Safe")
    label_2 = int(example["labels_2"] == "Safe")
    prompt_harmfulness = True if label_0 + label_1 + label_2 <= 1 else False

    return {
        "messages": [{"role": "user", "content": example["text"].strip()}],
        "labels": prompt_harmfulness,
        "annotations": [label_0, label_1, label_2],
    }


ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0", keep_in_memory=True)[
    "train"
]
ds = ds.map(
    convert_aegis,
    remove_columns=ds.column_names,
    desc="Converting Aegis dataset to the required format",
).filter(
    lambda x: x["messages"][0]["content"] != "",
)

ds.save_to_disk(SAVE_FOLDER / "aegis")


def convert_toxichat(example):
    return {
        "messages": [{"role": "user", "content": example["user_input"].strip()}],
        "labels": int(example["toxicity"] == 1),
        "jailbreaking": int(example["jailbreaking"] == 1),
    }


dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")["train"]

dataset = dataset.map(
    convert_toxichat,
    remove_columns=[x for x in dataset.column_names if x not in ["jailbreaking"]],
    desc="Converting ToxicChat dataset to the required format",
).filter(
    lambda x: x["messages"][0]["content"] != "",
)

dataset.save_to_disk(SAVE_FOLDER / "toxichat")


suicide_data = pd.read_csv("data/Suicide_Detection.csv", index_col=0)


suicide_data.head()


def convert_aegis_2(example):
    prompt_harmfulness = example["prompt_label"] == "unsafe"
    if example["prompt"] == "REDACTED":
        example["prompt"] = suicide_data["text"].loc[
            example["reconstruction_id_if_redacted"]
        ]
    return {
        "messages": [{"role": "user", "content": example["prompt"].strip()}],
        "labels": prompt_harmfulness,
    }


suicide_data.loc[256836]


ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", keep_in_memory=True)[
    "train"
]
ds = ds.filter(
    lambda x: x["prompt"].strip() != "",
)

redacted_ids = [i for i, x in enumerate(ds["prompt"]) if x == "REDACTED"]
ds = ds.map(
    convert_aegis_2,
    remove_columns=ds.column_names,
    desc="Converting Aegis dataset to the required format",
)

ds.save_to_disk(SAVE_FOLDER / "aegis2")

CLASS2LABEL = {
    "vanilla_benign": 0.0,
    "adversarial_benign": 0.0,
    "adversarial_harmful": 1.0,
    "vanilla_harmful": 1.0,
}


def convert_wildjailbreak(example):
    data = {}
    prompt = (
        example["vanilla"] if example["adversarial"] == "" else example["adversarial"]
    )
    data["messages"] = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example["completion"]},
    ]

    data["labels"] = CLASS2LABEL[example["data_type"]]
    return data


dataset = load_dataset(
    "allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False
)["train"].shuffle(seed=42).select(range(10000))

dataset = dataset.map(
    convert_wildjailbreak,
    remove_columns=dataset.column_names,
    desc="Converting WildJailbreak dataset to the required format",
).filter(
    lambda x: x["messages"][0]["content"] != "",
)
print(dataset)
dataset.save_to_disk(SAVE_FOLDER / "wildjailbreak")
