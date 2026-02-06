import json
from pathlib import Path

results_path = Path("results/prototypes_multilayer_different_c")


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

c_s = ["C_0.001", "C_0.005", "C_0.01", "C_0.05", "C_0.5", "C_1", "C_10"]

results_file_path = "harmful/wildguardtest/scores.json"

preds_file_path = "harmful/wildguardtest/predictions.parquet"

layers_used = {}
num_layers = {}
for folder in results_path.iterdir():
    layers_used[folder.name] = {}
    num_layers[folder.name] = None
    for c in c_s:
        with open((folder / c) / "meta_model_coefficients.json", "r") as f:
            meta_model_coefficients = json.load(f)
        max_coeff = max(abs(x) for x in meta_model_coefficients.values())
        layers_used[folder.name][c] = [
            abs(x) / max_coeff if max_coeff > 0 else 0
            for x in meta_model_coefficients.values()
        ]
        if num_layers[folder.name] is None:
            num_layers[folder.name] = len(meta_model_coefficients)

with open("plotting/data/layer_selection_data.json", "w") as f:
    json.dump({"num_layers": num_layers, "layers_used": layers_used}, f)
