import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

sys.path.append(".")

from itertools import combinations

from src.evaluation.utils import calculate_scores_multidataset
from src.prototype import (
    calculate_means_and_cov,
    get_bayes_precision_estimate,
    get_gda_params,
    get_gda_pred,
    get_mahalanobis_pred,
    get_nmc_pred,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(
    datapath: Path,
    num_samples_per_class: Optional[int] = None,
    random_state: int = 42,
    remove_outliers: bool = False,
):
    data = pd.read_parquet(datapath)
    if num_samples_per_class is not None:
        new_data = []
        for label in data["labels"].unique():
            new_data.append(
                data[data["labels"] == label].sample(
                    num_samples_per_class, random_state=random_state
                )
            )
        data = pd.concat(new_data).reset_index(drop=True)
    if remove_outliers:
        input_data = np.array(data["hidden_state"].tolist())
        print(f"Input data shape: {input_data.shape}")
        positive_clf = IsolationForest(contamination="auto")
        negative_clf = IsolationForest(contamination="auto")
        positive_clf.fit(input_data[data["labels"] == 1])
        negative_clf.fit(input_data[data["labels"] == 0])
        positive_pred = (positive_clf.predict(input_data) == 1) | (data["labels"] == 0)
        negative_pred = (negative_clf.predict(input_data) == 1) | (data["labels"] == 1)
        data = data[positive_pred & negative_pred]
        print(f"Data shape after removing outliers: {data.shape}")
        data = data.reset_index(drop=True)
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
):
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

    gda_scores = calculate_scores_multidataset(gda_scores, labels)
    nmc_scores = calculate_scores_multidataset(nmc_scores, labels)
    mahalanobis_scores = calculate_scores_multidataset(mahalanobis_scores, labels)

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
    wild_guard_hidden_states_folder=Path("wildgurad_train_hidden_states"),
    aegis_hidden_states_folder=Path("aegis_train_hidden_states"),
    toxichat_hidden_states_folder=Path("aegis_train_hidden_states"),
    layer_file_name: str = "layer_32_hidden_states.parquet",
    save_folder=Path("results/"),
    harmful_benchmarks_folder: Optional[Path] = None,
    non_harmful_benchmarks_folder: Optional[Path] = None,
    num_samples_per_class: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = False,
    add_means: bool = False,
):
    wildguard_hidden_states, wildguard_labels = load_data(
        wild_guard_hidden_states_folder / layer_file_name,
        num_samples_per_class=num_samples_per_class,
        random_state=random_state,
    )
    wildguard_means, wildguard_cov = calculate_means_and_cov(
        wildguard_hidden_states,
        labels=wildguard_labels,
    )
    aegis_hidden_states, aegis_labels = load_data(
        aegis_hidden_states_folder / layer_file_name,
        num_samples_per_class=num_samples_per_class,
        random_state=random_state,
        remove_outliers=False,
    )
    aegis_means, aegis_cov = calculate_means_and_cov(
        aegis_hidden_states,
        labels=aegis_labels,
    )
    toxichat_hidden_states, toxichat_labels = load_data(
        toxichat_hidden_states_folder / layer_file_name,
        num_samples_per_class=num_samples_per_class,
        random_state=random_state,
    )
    toxichat_means, toxichat_cov = calculate_means_and_cov(
        toxichat_hidden_states,
        labels=toxichat_labels,
    )

    covs = [wildguard_cov, aegis_cov, toxichat_cov]
    num_examples = [
        wildguard_hidden_states.size(0),
        aegis_hidden_states.size(0),
        toxichat_hidden_states.size(0),
    ]
    all_means = torch.concatenate([wildguard_means, aegis_means, toxichat_means], dim=0)
    datasets_names = [
        "wildguard",
        "aegis",
        "toxichat",
    ]
    all_inv_matrices = [
        get_bayes_precision_estimate(cov, n, wildguard_hidden_states.size(1))
        for cov, n in zip(covs, num_examples)
    ]

    for i in range(3):
        for selected_datasets in combinations(range(3), i + 1):
            cur_n = sum(num_examples[j] for j in selected_datasets)
            cur_cov = covs[selected_datasets[0]] * (
                num_examples[selected_datasets[0]] - 1
            )
            for j in selected_datasets[1:]:
                cur_cov = cur_cov + (covs[j] * (num_examples[j] - 1))
            cur_cov = cur_cov / (cur_n - 1)
            if add_means:
                means = (
                    all_means[2 * selected_datasets[0] : 2 * selected_datasets[0] + 2]
                    * num_examples[selected_datasets[0]]
                )
                for j in selected_datasets[1:]:
                    means = means + all_means[2 * j : 2 * j + 2] * num_examples[j]
                means = means / cur_n
                print(means.shape)
            else:
                means = []
                for j in selected_datasets:
                    means.append(all_means[2 * j : 2 * j + 2])
                means = torch.concatenate(means, dim=0)
            inv = get_bayes_precision_estimate(
                cur_cov, cur_n, wildguard_hidden_states.size(1)
            )
            inv_matrices = []
            for j in selected_datasets:
                inv_matrices += 2 * [all_inv_matrices[j]]
            inv_matrices = torch.stack(inv_matrices)
            gda_params = get_gda_params(means, inv)
            cur_dataset = "+".join([datasets_names[j] for j in selected_datasets])
            for harmful_benchmark in tqdm(list(harmful_benchmarks_folder.iterdir())):
                if verbose:
                    print(f"Processing harmful benchmark: {harmful_benchmark}")

                process_benchmark(
                    benchmark_path=harmful_benchmark,
                    gda_params=gda_params,
                    means=means,
                    inv_matrices=inv_matrices,
                    layer_file_name=layer_file_name,
                    save_folder=save_folder / cur_dataset / "harmful",
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
                        gda_params=gda_params,
                        means=means,
                        inv_matrices=inv_matrices,
                        layer_file_name=layer_file_name,
                        save_folder=save_folder / cur_dataset / "non_harmful",
                        verbose=verbose,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation hidden states.")
    parser.add_argument(
        "--wild_guard_hidden_states_folder",
        type=Path,
        help="Path to the WildGuard train dataset with calculated hidden_states.",
    )
    parser.add_argument(
        "--aegis_hidden_states_folder",
        type=Path,
        help="Path to the AEGIS train dataset with calculated hidden_states.",
    )
    parser.add_argument(
        "--toxichat_hidden_states_folder",
        type=Path,
        help="Path to the ToxicChat train dataset with calculated hidden_states.",
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
    parser.add_argument("--num_samples_per_class", type=int, default=None)
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random state for reproducibility."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")
    parser.add_argument(
        "--add_means",
        action="store_true",
        help="If set, add means instead concatenating them.",
    )
    args = parser.parse_args()

    main(
        wild_guard_hidden_states_folder=args.wild_guard_hidden_states_folder,
        aegis_hidden_states_folder=args.aegis_hidden_states_folder,
        toxichat_hidden_states_folder=args.toxichat_hidden_states_folder,
        layer_file_name=args.layer_file_name,
        save_folder=args.save_folder,
        harmful_benchmarks_folder=args.harmful_benchmarks_folder,
        non_harmful_benchmarks_folder=args.non_harmful_benchmarks_folder,
        num_samples_per_class=args.num_samples_per_class,
        random_state=args.random_state,
        verbose=args.verbose,
        add_means=args.add_means,
    )
