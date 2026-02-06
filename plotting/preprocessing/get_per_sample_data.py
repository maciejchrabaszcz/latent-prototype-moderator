import json
from pathlib import Path

import pandas as pd

results_path = Path("results/prototypes_multilayer_per_num_samples/")


results_file_path = "harmful/wildguardtest/scores.json"

preds_file_path = "harmful/wildguardtest/predictions.parquet"


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

results = []
for harmful_dataset in harmful_datasets:
    results_file_path = f"harmful/{harmful_dataset}/scores.json"
    for folder in results_path.iterdir():
        for seed in folder.iterdir():
            for num_samples in seed.iterdir():
                cur_results = num_samples / results_file_path
                if not cur_results.exists():
                    print(f"Skipping {folder}")
                    continue
                with open(cur_results, "r") as f:
                    data = json.load(f)
                data = {
                    "model": folder.name,
                    "num_samples": num_samples.name,
                    "seed": seed.name,
                    "dataset": harmful_dataset,
                    **data,
                }
                results.append(data)

        results_df = pd.DataFrame(results)

        results_df["num_samples"] = results_df["num_samples"].apply(
            lambda x: int(x.split("_")[0])
        )

tmp_results_df = results_df[
    results_df["model"].apply(lambda x: x.endswith("inst") or "qwen" in x)
]
tmp_results_df = tmp_results_df[tmp_results_df["num_samples"] >= 10]

tmp_results_df = tmp_results_df[tmp_results_df["dataset"] == "wildguardtest"]

tmp_results_df.to_csv("plotting/data/per_samples_data.csv", index=False)
