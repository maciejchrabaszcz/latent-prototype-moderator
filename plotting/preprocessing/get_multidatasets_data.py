import json
from pathlib import Path

import pandas as pd

results_path = Path("results/multilayer_prototypes_multidatasets/")

harmful_datasets = [
    "wildjailbreak_test",
    "simplesafety_tests",
    "toxicchat_humanannotated",
    "xstest",
    "aegis",
    "wildguardtest",
    "harmbench",
    "openai_mod",
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
datasets_of_interest = [
    "aegis",
    "toxicchat_humanannotated",
    "wildguardtest",
]
combinations_of_interest = [
    "wildguard",
    "wildguard+aegis",
    "wildguard+toxichat",
    "wildguard+aegis+toxichat",
]

names_of_combinations = {
    "wildguard": "WG",
    "wildguard+aegis": "WG + A",
    "wildguard+toxichat": "WG + TC",
    "wildguard+aegis+toxichat": "WG + A + TC",
}
models_names = {
    "llama3_8b_inst": "Llama3 8B",
    "mistral7b_inst": "Mistral 7B",
    "olmo2_1124_7b_inst": "OLMO2 7B",
    "qwen3_8b": "Qwen3 8B",
}
results = []
for model_folder in results_path.iterdir():
    for dataset in harmful_datasets + non_harmful_datasets:
        dataset_type = "harmful" if dataset in harmful_datasets else "non_harmful"
        results_file_path = f"{dataset_type}/{dataset}/scores.json"
        for folder in model_folder.iterdir():
            cur_results = folder / results_file_path
            if not cur_results.exists():
                print(f"Skipping {folder}")
                continue
            with open(cur_results, "r") as f:
                data = json.load(f)
            data = {
                "dataset": dataset,
                "model": model_folder.name,
                "num_datasets": folder.name,
                "dataset_type": dataset_type,
                "score": data["f1" if dataset_type == "harmful" else "accuracy"],
            }
            results.append(data)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "plotting/data/multilayer_multiple_datasets_plot_data.csv", index=False
)
