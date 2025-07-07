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

from src.evaluation.utils import calculate_scores
from src.prototype import (
    calculate_means_and_inv_cov,
    get_gda_params,
    get_gda_pred,
    get_mahalanobis_pred,
    get_nmc_pred,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(
    datapath: Path,
    num_samples_per_class: Optional[int] = None,
    random_state: int = None,
    num_frac_data: Optional[float] = None,
):
    data = pd.read_parquet(datapath)
    if num_frac_data is not None:
        data = data.sample(frac=num_frac_data, random_state=random_state)
    if num_samples_per_class is not None:
        new_data = []
        for label in data["labels"].unique():
            new_data.append(
                data[data["labels"] == label].sample(
                    num_samples_per_class, random_state=random_state
                )
            )
        data = pd.concat(new_data).reset_index(drop=True)
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

    gda_scores = calculate_scores(gda_scores, labels)
    nmc_scores = calculate_scores(nmc_scores, labels)
    mahalanobis_scores = calculate_scores(mahalanobis_scores, labels)

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
    num_samples_per_class: Optional[int] = None,
    num_frac_data: Optional[float] = None,
    random_state: int = 42,
    verbose: bool = False,
):
    print(layer_file_name)
    train_hidden_states, train_labels = load_data(
        train_hidden_states_folder / layer_file_name,
        num_samples_per_class=num_samples_per_class,
        random_state=random_state,
        num_frac_data=num_frac_data,
    )
    means, inv, inv_matrices = calculate_means_and_inv_cov(
        train_hidden_states,
        labels=train_labels,
        calculate_per_class_cov=True,
        scale_covariances=True,
    )
    gda_params = get_gda_params(means, inv)
    for harmful_benchmark in tqdm(list(harmful_benchmarks_folder.iterdir())):
        if verbose:
            print(f"Processing harmful benchmark: {harmful_benchmark}")

        process_benchmark(
            benchmark_path=harmful_benchmark,
            gda_params=gda_params,
            means=means,
            inv_matrices=inv_matrices,
            layer_file_name=layer_file_name,
            save_folder=save_folder / "harmful",
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
                save_folder=save_folder / "non_harmful",
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
    parser.add_argument("--num_samples_per_class", type=int, default=None)
    parser.add_argument("--num_frac_data", type=float, default=None)
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Random state for reproducibility.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")
    args = parser.parse_args()

    main(
        train_hidden_states_folder=args.train_hidden_states_folder,
        layer_file_name=args.layer_file_name,
        save_folder=args.save_folder,
        harmful_benchmarks_folder=args.harmful_benchmarks_folder,
        non_harmful_benchmarks_folder=args.non_harmful_benchmarks_folder,
        num_samples_per_class=args.num_samples_per_class,
        num_frac_data=args.num_frac_data,
        random_state=args.random_state,
        verbose=args.verbose,
    )
