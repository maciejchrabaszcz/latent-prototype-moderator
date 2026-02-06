import json
from pathlib import Path

import pandas as pd

results_path = Path("results/multilayer_prototypes_per_ordered_categories")
SAVE_PATH = Path("plotting/data/wgmix_categories_results.csv")
harmful_datasets = [
    "aegis",
    "harmbench",
    "openai_mod",
    "simplesafety_tests",
    "toxicchat_humanannotated",
    "wildguardtest",
    "wildjailbreak_test",
    "xstest",
]

non_harmful_datasets = [
    "mmlu",
    "codex",
    "gsm8k",
    "bbh",
    "mtbench",
    "truthfulqa",
    "alpaca_eval",
]
models_to_use = [
    "llama3_8b_inst",
    # "mistral12b_inst",
    "mistral7b_inst",
    # "olmo2_1124_13b_inst",
    "olmo2_1124_7b_inst",
    "qwen3_8b",
]

results = []
for model_folder in results_path.iterdir():
    for harmful_dataset in harmful_datasets:
        results_file_path = f"harmful/{harmful_dataset}/scores.json"
        for folder in model_folder.iterdir():
            cur_results = folder / results_file_path
            if not cur_results.exists():
                print(f"Skipping {folder}")
                continue
            with open(cur_results, "r") as f:
                data = json.load(f)
            data = {
                "num_categories": folder.name,
                "dataset": harmful_dataset,
                "dataset_type": "harmful",
                "model": model_folder.name,
                "score": data["f1"],
            }

            results.append(data)
results_df = pd.DataFrame(results)
results = []
for model_folder in results_path.iterdir():
    for non_harmful_dataset in non_harmful_datasets:
        results_file_path = f"non_harmful/{non_harmful_dataset}/scores.json"
        for folder in model_folder.iterdir():
            cur_results = folder / results_file_path
            if not cur_results.exists():
                print(f"Skipping {folder}")
                continue
            with open(cur_results, "r") as f:
                data = json.load(f)
            data = {
                "num_categories": folder.name,
                "dataset": non_harmful_dataset,
                "model": model_folder.name,
                "dataset_type": "non_harmful",
                "score": data["accuracy"],
            }
            results.append(data)

results_df = pd.concat([results_df, pd.DataFrame(results)], ignore_index=True)
results_df["num_categories"] = results_df["num_categories"].apply(
    lambda x: int(x.split("_")[0])
)
mean_df = (
    results_df[results_df.dataset == "wildguardtest"]
    .groupby(["model", "num_categories", "dataset_type"])
    .agg({"score": "mean"})
    .reset_index()
)

mean_df.to_csv(SAVE_PATH, index=False)
