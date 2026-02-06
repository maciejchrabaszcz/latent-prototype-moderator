import json
from pathlib import Path

import pandas as pd

results_path = Path("results/per_layer_prototypes")
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

results_file_path = "harmful/wildguardtest/scores.json"

preds_file_path = "harmful/wildguardtest/predictions.parquet"


gda_results = []
nmc_results = []
for folder in results_path.iterdir():
    for harmful_dataset in harmful_datasets:
        for layer_folder in folder.iterdir():
            cur_results = layer_folder / f"harmful/{harmful_dataset}/scores.json"
            if not cur_results.exists():
                print(f"Skipping {folder}")
                continue
            with open(cur_results, "r") as f:
                data = json.load(f)
            data["gda"] = {
                "model": folder.name,
                "dataset": harmful_dataset,
                "layer": layer_folder.name,
                **data["gda"],
            }
            data["nmc"] = {
                "model": folder.name,
                "dataset": harmful_dataset,
                "layer": layer_folder.name,
                **data["nmc"],
            }

            gda_results.append(data["gda"])
            nmc_results.append(data["nmc"])

gda_df = pd.DataFrame(gda_results)
nmc_df = pd.DataFrame(nmc_results)

gda_df["Classification_Type"] = "Hidden States Maha"
nmc_df["Classification_Type"] = "Hidden States NMC"

gda_df["layer"] = gda_df["layer"].apply(lambda x: int(x.split("_")[-1]))
nmc_df["layer"] = nmc_df["layer"].apply(lambda x: int(x.split("_")[-1]))


tmp_gda_df = gda_df[gda_df["model"].apply(lambda x: x.endswith("inst"))]


model_num_layers = {
    "llama3_8b_inst": 32,
    "mistral7b_inst": 32,
    "olmo2_1124_7b_inst": 32,
    "olmo2_1124_13b_inst": 40,
    "mistral12b_inst": 40,
    "olmo2_0325_32b_inst": 60,
}


tmp_gda_df["num_layers"] = tmp_gda_df["model"].map(model_num_layers)

tmp_gda_df.to_csv("plotting/data/per_layer_per_task_plotting_data.csv", index=False)
