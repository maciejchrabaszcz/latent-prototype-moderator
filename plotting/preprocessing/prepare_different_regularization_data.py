import json
from pathlib import Path

import pandas as pd

results_path = Path("results/prototypes_multilayer_different_c")
# results_path = Path(
#     "results/prototypes/per_layer/"
# )

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

c_s = ["C_0.0001", "C_0.001", "C_0.005", "C_0.01", "C_0.05", "C_0.5", "C_1", "C_10"]

results_file_path = "harmful/wildguardtest/scores.json"

preds_file_path = "harmful/wildguardtest/predictions.parquet"


results = []
for folder in results_path.iterdir():
    for c in c_s:
        with open((folder / c) / "meta_model_coefficients.json", "r") as f:
            meta_model_coefficients = json.load(f)
        num_layers_used = sum(
            [1 for coeff in meta_model_coefficients.values() if abs(coeff) > 1e-5]
        )
        for harmful_dataset in harmful_datasets:
            cur_results = folder / f"harmful/{harmful_dataset}/{c}/scores.json"
            if not cur_results.exists():
                print(f"Skipping {folder}")
                continue
            with open(cur_results, "r") as f:
                data = json.load(f)
            data = {
                "model": folder.name,
                "dataset": harmful_dataset,
                "C": c,
                "num_layers_used": num_layers_used,
                **data,
            }

            results.append(data)

results_df = pd.DataFrame(results)


results_df["C"] = results_df["C"].apply(lambda x: float(x.split("_")[-1]))


plot_df = (
    results_df.groupby(["model", "C", "num_layers_used"])["f1"]
    .agg("mean")
    .reset_index()
)
plot_df = plot_df[plot_df.num_layers_used >= 1]

plot_df.to_csv("plotting/data/different_regularization_data.csv", index=False)
