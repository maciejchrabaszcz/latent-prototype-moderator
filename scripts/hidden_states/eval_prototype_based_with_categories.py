import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(".")

from src.data.load_data import (
    load_wildguard_for_prompt_classification,
    load_wildguardtest_prompt,
)
from src.evaluation.utils import calculate_scores, calculate_scores_multiclass
from src.prototype import (
    calculate_means_and_inv_cov,
    get_gda_params,
    get_gda_pred,
    get_mahalanobis_pred,
    get_nmc_pred,
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


def load_train_data(
    datapath: Path,
):
    data = pd.read_parquet(datapath)
    categories = load_wildguard_for_prompt_classification()["category"]
    data["category"] = categories
    data["category"] = data["category"].map(CATEGORY_TO_LABEL)
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


def save_preds(gda_pred, nmc_pred, mahalanobis_preds, labels, save_path: Path):
    df = pd.DataFrame(
        {
            "gda_pred": gda_pred.cpu().numpy().tolist(),
            "nmc_pred": nmc_pred.cpu().numpy().tolist(),
            "mahalanobis_preds": mahalanobis_preds.cpu().numpy().tolist(),
            "labels": labels.cpu().numpy(),
        }
    )
    df.to_parquet(save_path / "predictions.parquet")


def save_scores(scores: dict, save_path: Path):
    with open(save_path / "scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4, ensure_ascii=True)


def process_benchmark(
    benchmark_path: Path,
    gda_params: dict,
    inv_matrices: torch.Tensor,
    means: torch.Tensor,
    layer_file_name: str,
    save_folder: Path,
    verbose: bool = False,
    hidden_states: Optional[torch.Tensor] = None,
    labels: Optional = None,
    binarize_output: bool = True,
    calculate_auc: bool = False,
    multiclass: bool = False,
):
    if hidden_states is None or labels is None:
        hidden_states, labels = load_data(benchmark_path / layer_file_name)
    gda_scores = get_gda_pred(hidden_states, **gda_params)
    nmc_scores = get_nmc_pred(hidden_states, means)
    mahalanobis_scores = get_mahalanobis_pred(hidden_states, means, inv_matrices)
    save_path = save_folder / benchmark_path.name
    save_path.mkdir(parents=True, exist_ok=True)

    save_preds(gda_scores, nmc_scores, mahalanobis_scores, labels, save_path)
    gda_scores = gda_scores.cpu().numpy()
    nmc_scores = nmc_scores.cpu().numpy()
    mahalanobis_scores = mahalanobis_scores.cpu().numpy()
    labels = labels.cpu().numpy()
    scores_fn = calculate_scores_multiclass if multiclass else calculate_scores

    gda_scores = scores_fn(
        gda_scores, labels, binarize_output=binarize_output, calculate_auc=calculate_auc
    )
    nmc_scores = scores_fn(
        nmc_scores, labels, binarize_output=binarize_output, calculate_auc=calculate_auc
    )
    mahalanobis_scores = scores_fn(
        mahalanobis_scores,
        labels,
        binarize_output=binarize_output,
        calculate_auc=calculate_auc,
    )

    all_scores = {
        "gda": gda_scores,
        "nmc": nmc_scores,
        "mahalanobis": mahalanobis_scores,
    }
    if verbose:
        print(f"Scores for {benchmark_path.name}:")
        print(all_scores)
    save_scores(all_scores, save_path)


def main(
    train_hidden_states_folder=Path("wildgurad_train_hidden_states"),
    layer_file_name: str = "layer_32_hidden_states.parquet",
    save_folder=Path("results/"),
    harmful_benchmarks_folder: Optional[Path] = None,
    non_harmful_benchmarks_folder: Optional[Path] = None,
    verbose: bool = False,
):
    data = load_train_data(
        train_hidden_states_folder / layer_file_name,
    )
    test_data = load_wildguard_test_data(
        harmful_benchmarks_folder / "wildguardtest" / layer_file_name,
    )
    categories_of_interest = [0]
    num_categories = 1
    for i in range(1, len(CATEGORY_TO_LABEL)):
        cur_save_folder = save_folder / f"{i}_categories"
        categories_of_interest.append(i)
        num_categories += 1
        with torch.no_grad():
            cur_data = data[data["category"].isin(categories_of_interest)]
            train_hidden_states = torch.tensor(
                np.array(cur_data["hidden_state"].tolist())
            ).to(DEVICE)
            train_labels = torch.tensor(np.array(cur_data["category"].tolist())).to(DEVICE)

            cur_test_data = test_data[test_data["category"].isin(categories_of_interest)]
            test_hidden_states = torch.tensor(
                np.array(cur_test_data["hidden_state"].tolist())
            ).to(DEVICE)
            test_labels = torch.tensor(np.array(cur_test_data["category"].tolist())).to(
                DEVICE
            )

            means, inv, inv_matrices = calculate_means_and_inv_cov(
                train_hidden_states,
                labels=train_labels,
                calculate_per_class_cov=True,
                scale_covariances=True,
            )
            gda_params = get_gda_params(means, inv)

            process_benchmark(
                Path("wildguardtest_per_category"),
                gda_params=gda_params,
                inv_matrices=inv_matrices,
                means=means,
                layer_file_name=layer_file_name,
                save_folder=cur_save_folder,
                hidden_states=test_hidden_states,
                labels=test_labels,
                verbose=verbose,
                binarize_output=True,
                calculate_auc=True,
                multiclass=num_categories > 2,
            )
            for harmful_benchmark in tqdm(list(harmful_benchmarks_folder.iterdir())):
                if verbose:
                    print(f"Processing harmful benchmark: {harmful_benchmark}")

                process_benchmark(
                    benchmark_path=harmful_benchmark,
                    gda_params=gda_params,
                    means=means,
                    inv_matrices=inv_matrices,
                    layer_file_name=layer_file_name,
                    save_folder=cur_save_folder / "harmful",
                    verbose=verbose,
                )
            if non_harmful_benchmarks_folder is not None:
                for non_harmful_benchmark in tqdm(
                    list(non_harmful_benchmarks_folder.iterdir())
                ):
                    if verbose:
                        print(f"Processing non-harmful benchmark: {non_harmful_benchmark}")

                    process_benchmark(
                        benchmark_path=non_harmful_benchmark,
                        gda_params=gda_params,
                        means=means,
                        inv_matrices=inv_matrices,
                        layer_file_name=layer_file_name,
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
        "--layer_file_name",
        type=str,
        default="layer_32_hidden_states.parquet",
        help="Name of the layer file to process.",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default=Path("hidden_states_results"),
        help="Folder to save hidden states.",
    )
    parser.add_argument("--harmful_benchmarks_folder", type=Path, default=None)
    parser.add_argument("--non_harmful_benchmarks_folder", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")
    args = parser.parse_args()

    main(
        train_hidden_states_folder=args.train_hidden_states_folder,
        layer_file_name=args.layer_file_name,
        save_folder=args.save_folder,
        harmful_benchmarks_folder=args.harmful_benchmarks_folder,
        non_harmful_benchmarks_folder=args.non_harmful_benchmarks_folder,
        verbose=args.verbose,
    )
