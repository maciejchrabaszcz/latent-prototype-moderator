import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

sys.path.append(".")

from src.data.load_data import (
    load_wildguard_for_prompt_classification,
    load_wildguardtest_prompt,
)
from src.evaluation.utils import calculate_scores
from src.prototype import (
    calculate_means_and_inv_cov,
    get_gda_params,
    get_gda_pred,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CATEGORIES = [
#     "benign",
#     "causing_material_harm_by_disseminating_misinformation",
#     "copyright_violations",
#     "cyberattack",
#     "defamation_encouraging_unethical_or_unsafe_actions",
#     "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns",
#     "fraud_assisting_illegal_activities",
#     "mental_health_over-reliance_crisis",
#     "others",
#     "private_information_individual",
#     "sensitive_information_organization_government",
#     "sexual_content",
#     "social_stereotypes_and_unfair_discrimination",
#     "toxic_language_hate_speech",
#     "violence_and_physical_harm",
# ]
# Ordered by counts
CATEGORIES = [
    "benign",
    "others",
    "social_stereotypes_and_unfair_discrimination",
    "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns",
    "sensitive_information_organization_government",
    "toxic_language_hate_speech",
    "violence_and_physical_harm",
    "private_information_individual",
    "defamation_encouraging_unethical_or_unsafe_actions",
    "fraud_assisting_illegal_activities",
    "sexual_content",
    "mental_health_over-reliance_crisis",
    "copyright_violations",
    "cyberattack",
    "causing_material_harm_by_disseminating_misinformation",
]
CATEGORY_TO_LABEL = {x: i for i, x in enumerate(CATEGORIES)}


def load_train_data(datapath: Path, categories=None):
    data = pd.read_parquet(datapath)
    if categories is None:
        categories = load_wildguard_for_prompt_classification()["category"]
        data["category"] = categories
        data["category"] = data["category"].map(CATEGORY_TO_LABEL)
    else:
        data["category"] = categories
    return data


def load_wildguard_test_data(datapath: Path):
    data = pd.read_parquet(datapath)
    categories = load_wildguardtest_prompt()["subcategory"]
    data["category"] = categories
    data["category"] = data["category"].map(CATEGORY_TO_LABEL)
    return data


def load_data(
    datapath: Path,
):
    data = pd.read_parquet(datapath)
    hidden_states = torch.tensor(np.array(data["hidden_state"].tolist())).to(DEVICE)
    labels = torch.tensor(np.array(data["labels"].tolist())).to(DEVICE)
    return hidden_states, labels


def save_preds(preds, labels, save_path: Path):
    df = pd.DataFrame(
        {
            "preds": preds.tolist(),
            "labels": labels.cpu().numpy(),
        }
    )
    df.to_parquet(save_path / "predictions.parquet")


def save_scores(scores: dict, save_path: Path):
    with open(save_path / "scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4, ensure_ascii=True)


def calculate_multilayer_scores(
    all_gda_params, benchmark_path: Path, meta_model: LogisticRegression
):
    all_preds = {}
    for layer_file_name, gda_params in all_gda_params.items():
        hidden_states, labels = load_data(benchmark_path / layer_file_name)
        preds = get_gda_pred(hidden_states, **gda_params)
        preds = preds[:, list(range(1, preds.size(1), 2))].sum(dim=1, keepdim=False)
        all_preds[layer_file_name] = preds.detach().cpu().numpy()
    meta_df = pd.DataFrame(all_preds)

    meta_preds = meta_model.predict_proba(meta_df)[:, 1]

    return meta_preds, labels.cpu().numpy()


def process_benchmark(
    benchmark_path: Path,
    all_gda_params: dict,
    meta_model: LogisticRegression,
    save_folder: Path,
    verbose: bool = False,
):
    all_preds = {}
    save_path = save_folder / benchmark_path.name
    save_path.mkdir(parents=True, exist_ok=True)

    for layer_file_name, gda_params in all_gda_params.items():
        hidden_states, labels = load_data(benchmark_path / layer_file_name)
        preds = get_gda_pred(hidden_states, **gda_params)
        preds = preds[:, 1:].detach().cpu().numpy()
        for i in range(preds.shape[1]):
            all_preds[layer_file_name + f"_{i}"] = preds[:, i]
    meta_df = pd.DataFrame(all_preds)

    meta_preds = meta_model.predict_proba(meta_df)[:, 1]
    all_scores = calculate_scores(meta_preds, labels.cpu().numpy())
    save_preds(meta_preds, labels, save_path)
    if verbose:
        print(f"Scores for {benchmark_path.name}:")
        print(all_scores)
    save_scores(all_scores, save_path)


def main(
    train_hidden_states_folder=Path("wildgurad_train_hidden_states"),
    save_folder=Path("results/"),
    harmful_benchmarks_folder: Optional[Path] = None,
    non_harmful_benchmarks_folder: Optional[Path] = None,
    finetune_meta_model: bool = True,
    verbose: bool = False,
):
    ### TODO Make it so that for each category we calculate prototypes and use preds from each prototype as input to meta model
    meta_model = None
    layers_to_process = [
        x for x in train_hidden_states_folder.iterdir() if x.name.split("_")[1] != "0"
    ]
    all_gda_params = {i: {} for i in range(1, len(CATEGORY_TO_LABEL))}
    all_train_preds = {i: {} for i in range(1, len(CATEGORY_TO_LABEL))}
    all_train_labels = {i: None for i in range(1, len(CATEGORY_TO_LABEL))}
    all_meta_models = {i: None for i in range(1, len(CATEGORY_TO_LABEL))}
    categories = None
    for layer_file in tqdm(layers_to_process, desc="Processing layers"):
        data = load_train_data(layer_file, categories=categories)
        if categories is None:
            categories = data["category"]
        categories_of_interest = [0]
        with torch.no_grad():
            for i in tqdm(
                range(1, len(CATEGORY_TO_LABEL)), desc="Processing categories", leave=False
            ):
                categories_of_interest.append(i)
                cur_data = data[data["category"].isin(categories_of_interest)]
                train_hidden_states = torch.tensor(
                    np.array(cur_data["hidden_state"].tolist())
                ).to(DEVICE)
                train_labels = torch.tensor(np.array(cur_data["category"].tolist())).to(
                    DEVICE
                )
                means, inv = calculate_means_and_inv_cov(
                    train_hidden_states,
                    labels=train_labels,
                    calculate_per_class_cov=False,
                    scale_covariances=False,
                )
                gda_params = get_gda_params(means, inv)
                all_gda_params[i][layer_file.name] = gda_params
                preds = (get_gda_pred(train_hidden_states, **gda_params)
                    .detach()
                    .cpu()
                    .numpy()[:, 1:])
                if all_train_labels[i] is None:
                    all_train_labels[i] = (train_labels >= 1).cpu().numpy()
                for j in range(preds.shape[1]):
                    all_train_preds[i][layer_file.name + f"_{j}"] = (
                        preds[:, j]
                    )
    for i in range(1, len(CATEGORY_TO_LABEL)):
        if meta_model is None or finetune_meta_model:
            meta_df = pd.DataFrame(all_train_preds[i])
            meta_model = LogisticRegression(
                C=0.01, penalty="l1", solver="liblinear"
            )
            meta_model.fit(
                meta_df, all_train_labels[i]
            )
        all_meta_models[i] = meta_model

    for i in tqdm(range(1, len(CATEGORY_TO_LABEL)), desc="Processing results for categories"):
        cur_save_folder = save_folder / f"{i}_categories"
        with torch.no_grad():
            for harmful_benchmark in tqdm(list(harmful_benchmarks_folder.iterdir())):
                if verbose:
                    print(f"Processing harmful benchmark: {harmful_benchmark}")

                process_benchmark(
                    benchmark_path=harmful_benchmark,
                    all_gda_params=all_gda_params[i],
                    meta_model=all_meta_models[i],
                    save_folder=cur_save_folder / "harmful",
                    verbose=verbose,
                )
            if non_harmful_benchmarks_folder is not None:
                for non_harmful_benchmark in tqdm(
                    list(non_harmful_benchmarks_folder.iterdir())
                ):
                    if verbose:
                        print(
                            f"Processing non-harmful benchmark: {non_harmful_benchmark}"
                        )

                    process_benchmark(
                        benchmark_path=non_harmful_benchmark,
                        all_gda_params=all_gda_params[i],
                        meta_model=all_meta_models[i],
                        save_folder=cur_save_folder / "non_harmful",
                        verbose=verbose,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation hidden states.")
    parser.add_argument(
        "--train_hidden_states_folder",
        type=Path,
        help="Path to the WildGuard train dataset with calculated hidden_states.",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default=Path("hidden_states_results"),
        help="Folder to save hidden states.",
    )
    parser.add_argument("--harmful_benchmarks_folder", type=Path, default=None)
    parser.add_argument("--non_harmful_benchmarks_folder", type=Path, default=None)
    parser.add_argument(
        "--finetune_meta_model", action="store_true", help="Finetune meta model."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")
    args = parser.parse_args()

    main(
        train_hidden_states_folder=args.train_hidden_states_folder,
        save_folder=args.save_folder,
        harmful_benchmarks_folder=args.harmful_benchmarks_folder,
        non_harmful_benchmarks_folder=args.non_harmful_benchmarks_folder,
        finetune_meta_model=True,
        verbose=args.verbose,
    )
