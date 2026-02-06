import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
sys.path.append(".")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.evaluation.utils import calculate_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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
    hidden_states = np.array(data["hidden_state"].tolist())
    labels = np.array(data["labels"].tolist())
    return hidden_states, labels


def save_preds(lr_pred, rf_pred, xgb_pred, labels, save_path: Path):
    df = pd.DataFrame(
        {
            "lr_pred": lr_pred.tolist(),
            "rf_pred": rf_pred.tolist(),
            "xgb_pred": xgb_pred.tolist(),
            "labels": labels.tolist(),
        }
    )
    df.to_parquet(save_path / "predictions.parquet")


def save_scores(scores: dict, save_path: Path):
    with open(save_path / "scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4, ensure_ascii=True)


def fit_models(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
):
    logging.info("Fitting Logistic Regression model...")
    lr_model = LogisticRegression(n_jobs=-1, random_state=random_state)
    lr_model.fit(hidden_states, labels)

    logging.info("Fitting Random Forest model...")
    rf_model = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    rf_model.fit(hidden_states, labels)

    logging.info("Fitting XGBoost model...")
    xgb_model = XGBClassifier(n_jobs=-1, random_state=random_state)
    xgb_model.fit(hidden_states, labels)

    return lr_model, rf_model, xgb_model


def get_preds(
    hidden_states: np.ndarray,
    lr_model: LogisticRegression,
    rf_model: RandomForestClassifier,
    xgb_model: XGBClassifier,
):
    lr_pred = lr_model.predict_proba(hidden_states)
    rf_pred = rf_model.predict_proba(hidden_states)
    xgb_pred = xgb_model.predict_proba(hidden_states)

    return lr_pred, rf_pred, xgb_pred


def process_benchmark(
    benchmark_path: Path,
    lr_model: LogisticRegression,
    rf_model: RandomForestClassifier,
    xgb_model: XGBClassifier,
    layer_file_name: str,
    save_folder: Path,
    verbose: bool = False,
):
    hidden_states, labels = load_data(benchmark_path / layer_file_name)
    save_path = save_folder / benchmark_path.name
    save_path.mkdir(parents=True, exist_ok=True)
    lr_pred, rf_pred, xgb_pred = get_preds(
        hidden_states,
        lr_model,
        rf_model,
        xgb_model,
    )

    save_preds(lr_pred, rf_pred, xgb_pred, labels, save_path)

    lr_pred = calculate_scores(lr_pred, labels)
    rf_pred = calculate_scores(rf_pred, labels)
    xgb_pred = calculate_scores(xgb_pred, labels)

    all_scores = {
        "lr": lr_pred,
        "rf": rf_pred,
        "xgb": xgb_pred,
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
    rf_model, lr_model, xgb_model = fit_models(
        train_hidden_states,
        train_labels,
        random_state=random_state,
    )
    for harmful_benchmark in tqdm(list(harmful_benchmarks_folder.iterdir())):
        if verbose:
            print(f"Processing harmful benchmark: {harmful_benchmark}")

        process_benchmark(
            benchmark_path=harmful_benchmark,
            lr_model=lr_model,
            rf_model=rf_model,
            xgb_model=xgb_model,
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
                lr_model=lr_model,
                rf_model=rf_model,
                xgb_model=xgb_model,
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
