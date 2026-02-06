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

from src.evaluation.utils import calculate_scores
from src.prototype import (
    calculate_means_and_inv_cov,
    get_gda_params,
    get_gda_pred,
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


def process_benchmark(
    benchmark_path: Path,
    all_gda_params: dict,
    meta_model: LogisticRegression,
    save_folder: Path,
    use_nmc: bool = False,
    use_distance: bool = False,
    verbose: bool = False,
):
    all_preds = {}
    save_path = save_folder / benchmark_path.name
    save_path.mkdir(parents=True, exist_ok=True)

    for layer_file_name, gda_params in all_gda_params.items():
        hidden_states, labels = load_data(benchmark_path / layer_file_name)
        if use_nmc:
            preds = get_nmc_pred(
                hidden_states,
                gda_params,
                return_probs=not use_distance,
                return_logits=use_distance,
            )
        else:
            preds = get_gda_pred(
                hidden_states,
                return_probs=not use_distance,
                return_logits=use_distance,
                **gda_params,
            )
        all_preds[layer_file_name] = preds.detach().cpu().numpy()[:, 1]
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
    num_samples_per_class: Optional[int] = None,
    num_frac_data: Optional[float] = None,
    random_state: int = 42,
    use_nmc: bool = False,
    use_distance: bool = False,
    verbose: bool = False,
):
    all_gda_params = {}
    all_train_preds = {}
    layers_to_process = [
        x for x in train_hidden_states_folder.iterdir() if x.name.split("_")[1] != "0"
    ]
    for layer_file in tqdm(layers_to_process, desc="Processing layers"):
        train_hidden_states, train_labels = load_data(
            layer_file,
            num_samples_per_class=num_samples_per_class,
            random_state=random_state,
            num_frac_data=num_frac_data,
        )
        means, inv, _ = calculate_means_and_inv_cov(
            train_hidden_states,
            labels=train_labels,
            calculate_per_class_cov=True,
            scale_covariances=True,
        )
        gda_params = get_gda_params(means, inv)
        all_gda_params[layer_file.name] = gda_params if not use_nmc else means
        if use_nmc:
            all_train_preds[layer_file.name] = (
                get_nmc_pred(
                    train_hidden_states,
                    means,
                    return_probs=not use_distance,
                    return_logits=use_distance,
                )
                .detach()
                .cpu()
                .numpy()
            )[:, 1]
        else:
            all_train_preds[layer_file.name] = (
                get_gda_pred(
                    train_hidden_states,
                    return_probs=not use_distance,
                    return_logits=use_distance,
                    **gda_params,
                )
                .detach()
                .cpu()
                .numpy()
            )[:, 1]
    meta_df = pd.DataFrame(all_train_preds)
    meta_model = LogisticRegression(C=1.0, penalty="l1", solver="liblinear")
    meta_model.fit(meta_df, train_labels.cpu().numpy())
    # save coefficients
    save_folder.mkdir(parents=True, exist_ok=True)
    with open(save_folder / "meta_model_coefficients.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                layer_id: coef
                for layer_id, coef in zip(meta_df.columns, meta_model.coef_[0].tolist())
            },
            f,
            indent=4,
            ensure_ascii=True,
        )

    for harmful_benchmark in tqdm(list(harmful_benchmarks_folder.iterdir())):
        if verbose:
            print(f"Processing harmful benchmark: {harmful_benchmark}")

        process_benchmark(
            benchmark_path=harmful_benchmark,
            all_gda_params=all_gda_params,
            meta_model=meta_model,
            save_folder=save_folder / "harmful",
            use_nmc=use_nmc,
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
                all_gda_params=all_gda_params,
                meta_model=meta_model,
                save_folder=save_folder / "non_harmful",
                use_nmc=use_nmc,
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
    parser.add_argument("--num_samples_per_class", type=int, default=None)
    parser.add_argument("--num_frac_data", type=float, default=None)
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Random state for reproducibility.",
    )
    parser.add_argument("--use_nmc", action="store_true", help="Enable NMC.")
    parser.add_argument(
        "--use_distance", action="store_true", help="Use distance-based features."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")
    args = parser.parse_args()

    main(
        train_hidden_states_folder=args.train_hidden_states_folder,
        save_folder=args.save_folder,
        harmful_benchmarks_folder=args.harmful_benchmarks_folder,
        non_harmful_benchmarks_folder=args.non_harmful_benchmarks_folder,
        num_samples_per_class=args.num_samples_per_class,
        num_frac_data=args.num_frac_data,
        random_state=args.random_state,
        use_nmc=args.use_nmc,
        use_distance=args.use_distance,
        verbose=args.verbose,
    )
