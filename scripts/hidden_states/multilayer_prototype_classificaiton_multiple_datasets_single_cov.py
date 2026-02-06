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
    calculate_means_and_cov,
    get_bayes_precision_estimate,
    get_gda_params,
    get_gda_pred,
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
    all_gda_params: list[dict],
    meta_model: LogisticRegression,
    save_folder: Path,
    verbose: bool = False,
):
    all_preds = {}
    save_path = save_folder / benchmark_path.name
    save_path.mkdir(parents=True, exist_ok=True)
    all_preds = {}
    layer_file_names = list(all_gda_params[0].keys())
    for layer_file_name in layer_file_names:
        hidden_states, labels = load_data(benchmark_path / layer_file_name)
        for i, gda_params in enumerate(all_gda_params):
            preds = get_gda_pred(hidden_states, **gda_params[layer_file_name])
            preds = preds[:, 1]
            all_preds[layer_file_name + f"{i}"] = preds.detach().cpu().numpy()

    meta_df = pd.DataFrame(all_preds)
    meta_preds = meta_model.predict_proba(meta_df)[:, 1]

    all_scores = calculate_scores(meta_preds, labels.cpu().numpy())
    save_preds(meta_preds, labels, save_path)
    if verbose:
        print(f"Scores for {benchmark_path.name}:")
        print(all_scores)
    save_scores(all_scores, save_path)


def get_gda_params_for_all_layers(
    all_params: dict,
    num_examples: list[int],
    selected_datasets: tuple[int],
):
    all_gda_params = {}

    cur_n = sum(num_examples[j] for j in selected_datasets)
    for layer_file_name, params in all_params.items():
        covs = params["covs"]
        all_means = params["means"]
        hidden_state_dim = params["hidden_state_dim"]
        cur_cov = covs[selected_datasets[0]] * (num_examples[selected_datasets[0]] - 1)
        for j in selected_datasets[1:]:
            cur_cov = cur_cov + (covs[j] * (num_examples[j] - 1))
        cur_cov = cur_cov / (cur_n - 1)
        means = []
        for j in selected_datasets:
            means.append(all_means[2 * j : 2 * j + 2])
        means = torch.concatenate(means, dim=0)
        inv = get_bayes_precision_estimate(cur_cov, cur_n, hidden_state_dim)

        all_gda_params[layer_file_name] = get_gda_params(means, inv)
    return all_gda_params


def main(
    wild_guard_hidden_states_folder=Path("wildgurad_train_hidden_states"),
    aegis_hidden_states_folder=Path("aegis_train_hidden_states"),
    toxichat_hidden_states_folder=Path("aegis_train_hidden_states"),
    save_folder=Path("results/"),
    harmful_benchmarks_folder: Optional[Path] = None,
    non_harmful_benchmarks_folder: Optional[Path] = None,
    num_samples_per_class: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = False,
):
    all_wg_train_preds = {}
    all_wg_aegis_train_preds = {}
    all_wg_toxichat_train_preds = {}
    all_wg_aegis_toxichat_train_preds = {}
    all_wg_gda_params = {}
    all_wg_aegis_gda_params = [{}, {}]
    all_wg_toxichat_gda_params = [{}, {}]
    all_wg_aegis_toxichat_gda_params = [{}, {}, {}]
    layers_to_process = [
        x
        for x in wild_guard_hidden_states_folder.iterdir()
        if x.name.split("_")[1] != "0"
    ]
    for layer_file in tqdm(layers_to_process, desc="Processing layers"):
        wildguard_hidden_states, wildguard_labels = load_data(
            layer_file,
            num_samples_per_class=num_samples_per_class,
            random_state=random_state,
        )
        wg_means, wg_cov = calculate_means_and_cov(
            wildguard_hidden_states,
            labels=wildguard_labels,
        )
        inv = get_bayes_precision_estimate(
            wg_cov, wildguard_hidden_states.size(0), wildguard_hidden_states.size(1)
        )
        wg_gda_params = get_gda_params(wg_means, inv)
        all_wg_gda_params[layer_file.name] = wg_gda_params
        all_wg_train_preds[layer_file.name + "0"] = (
            get_gda_pred(wildguard_hidden_states, **wg_gda_params)
            .detach()
            .cpu()
            .numpy()[:, 1]
        )

        aegis_hidden_states, aegis_labels = load_data(
            aegis_hidden_states_folder / layer_file.name,
            num_samples_per_class=num_samples_per_class,
            random_state=random_state,
            remove_outliers=False,
        )
        aegis_means, aegis_cov = calculate_means_and_cov(
            aegis_hidden_states,
            labels=aegis_labels,
        )
        wg_aegis_hidden_states = torch.concatenate(
            [wildguard_hidden_states, aegis_hidden_states], dim=0
        )
        cov = (
            wg_cov * (wildguard_hidden_states.size(0) - 1)
            + aegis_cov * (aegis_hidden_states.size(0) - 1)
        ) / (wg_aegis_hidden_states.size(0) - 1)
        inv = get_bayes_precision_estimate(
            cov, wg_aegis_hidden_states.size(0), wg_aegis_hidden_states.size(1)
        )
        aegis_gda_params = get_gda_params(aegis_means, inv)
        wg_aegis_gda_params = get_gda_params(wg_means, inv)
        wg_aegis_labels = torch.concatenate([wildguard_labels, aegis_labels], dim=0)
        all_wg_aegis_gda_params[0][layer_file.name] = wg_aegis_gda_params
        all_wg_aegis_gda_params[1][layer_file.name] = aegis_gda_params
        all_wg_aegis_train_preds[layer_file.name + "0"] = (
            get_gda_pred(wg_aegis_hidden_states, **wg_aegis_gda_params)
            .detach()
            .cpu()
            .numpy()[:, 1]
        )

        all_wg_aegis_train_preds[layer_file.name + "1"] = (
            get_gda_pred(wg_aegis_hidden_states, **aegis_gda_params)
            .detach()
            .cpu()
            .numpy()[:, 1]
        )

        toxichat_hidden_states, toxichat_labels = load_data(
            toxichat_hidden_states_folder / layer_file.name,
            num_samples_per_class=num_samples_per_class,
            random_state=random_state,
        )
        toxichat_means, toxichat_cov = calculate_means_and_cov(
            toxichat_hidden_states,
            labels=toxichat_labels,
        )
        wg_toxichat_hidden_states = torch.concatenate(
            [wildguard_hidden_states, toxichat_hidden_states], dim=0
        )
        wg_toxichat_labels = torch.concatenate(
            [wildguard_labels, toxichat_labels], dim=0
        )
        cov = (
            wg_cov * (wildguard_hidden_states.size(0) - 1)
            + toxichat_cov * (toxichat_hidden_states.size(0) - 1)
        ) / (wg_toxichat_hidden_states.size(0) - 1)
        inv = get_bayes_precision_estimate(
            cov, wg_toxichat_hidden_states.size(0), wg_toxichat_hidden_states.size(1)
        )
        toxichat_gda_params = get_gda_params(toxichat_means, inv)
        wg_toxichat_gda_params = get_gda_params(wg_means, inv)
        all_wg_toxichat_gda_params[0][layer_file.name] = wg_toxichat_gda_params
        all_wg_toxichat_gda_params[1][layer_file.name] = toxichat_gda_params

        all_wg_toxichat_train_preds[layer_file.name + "0"] = (
            get_gda_pred(wg_toxichat_hidden_states, **wg_toxichat_gda_params)
            .detach()
            .cpu()
            .numpy()[:, 1]
        )
        all_wg_toxichat_train_preds[layer_file.name + "1"] = (
            get_gda_pred(wg_toxichat_hidden_states, **toxichat_gda_params)
            .detach()
            .cpu()
            .numpy()[:, 1]
        )
        wg_aegis_toxichat_hidden_states = torch.concatenate(
            [wildguard_hidden_states, aegis_hidden_states, toxichat_hidden_states],
            dim=0,
        )
        wg_aegis_toxichat_labels = torch.concatenate(
            [wildguard_labels, aegis_labels, toxichat_labels], dim=0
        )
        cov = (
            wg_cov * (wildguard_hidden_states.size(0) - 1)
            + aegis_cov * (aegis_hidden_states.size(0) - 1)
            + toxichat_cov * (toxichat_hidden_states.size(0) - 1)
        ) / (wg_aegis_toxichat_hidden_states.size(0) - 1)
        inv = get_bayes_precision_estimate(
            cov,
            wg_aegis_toxichat_hidden_states.size(0),
            wg_aegis_toxichat_hidden_states.size(1),
        )
        toxichat_gda_params = get_gda_params(toxichat_means, inv)
        aegis_gda_params = get_gda_params(aegis_means, inv)
        wg_aegis_toxichat_gda_params = get_gda_params(wg_means, inv)
        all_wg_aegis_toxichat_gda_params[0][layer_file.name] = (
            wg_aegis_toxichat_gda_params
        )
        all_wg_aegis_toxichat_gda_params[1][layer_file.name] = aegis_gda_params
        all_wg_aegis_toxichat_gda_params[2][layer_file.name] = toxichat_gda_params
        all_wg_aegis_toxichat_train_preds[layer_file.name + "0"] = (
            get_gda_pred(
                wg_aegis_toxichat_hidden_states, **wg_aegis_toxichat_gda_params
            )
            .detach()
            .cpu()
            .numpy()[:, 1]
        )
        all_wg_aegis_toxichat_train_preds[layer_file.name + "1"] = (
            get_gda_pred(wg_aegis_toxichat_hidden_states, **aegis_gda_params)
            .detach()
            .cpu()
            .numpy()[:, 1]
        )
        all_wg_aegis_toxichat_train_preds[layer_file.name + "2"] = (
            get_gda_pred(
                wg_aegis_toxichat_hidden_states, **toxichat_gda_params
            )
            .detach()
            .cpu()
            .numpy()[:, 1]
        )
    combinations_of_datasets = [
        (0,),
        (0, 1),
        (0, 2),
        (0, 1, 2),
    ]
    datasets_names = [
        "wildguard",
        "aegis",
        "toxichat",
    ]
    meta_wg_df = pd.DataFrame(all_wg_train_preds)
    meta_wg_aegis_df = pd.DataFrame(all_wg_aegis_train_preds)
    meta_wg_toxichat_df = pd.DataFrame(all_wg_toxichat_train_preds)
    meta_wg_aegis_toxichat_df = pd.DataFrame(all_wg_aegis_toxichat_train_preds)
    meta_models = [
        LogisticRegression(C=0.01, penalty="l1", solver="liblinear").fit(
            meta_wg_df, wildguard_labels.cpu().numpy()
        ),
        LogisticRegression(C=0.01, penalty="l1", solver="liblinear").fit(
            meta_wg_aegis_df, wg_aegis_labels.cpu().numpy()
        ),
        LogisticRegression(C=0.01, penalty="l1", solver="liblinear").fit(
            meta_wg_toxichat_df, wg_toxichat_labels.cpu().numpy()
        ),
        LogisticRegression(C=0.01, penalty="l1", solver="liblinear").fit(
            meta_wg_aegis_toxichat_df, wg_aegis_toxichat_labels.cpu().numpy()
        ),
    ]
    all_gda_params = [
        [all_wg_gda_params],
        all_wg_aegis_gda_params,
        all_wg_toxichat_gda_params,
        all_wg_aegis_toxichat_gda_params,
    ]
    # for i in range(3):
    # for selected_datasets in combinations(range(3), i + 1):
    for i, (selected_datasets, cur_gda_params) in enumerate(
        zip(combinations_of_datasets, all_gda_params)
    ):
        # all_gda_params = get_gda_params_for_all_layers(
        #     all_params=all_params,
        #     num_examples=num_examples,
        #     selected_datasets=selected_datasets,
        # )
        cur_meta_model = meta_models[i]
        cur_dataset = "+".join([datasets_names[j] for j in selected_datasets])
        for harmful_benchmark in tqdm(list(harmful_benchmarks_folder.iterdir())):
            if verbose:
                print(f"Processing harmful benchmark: {harmful_benchmark}")

            process_benchmark(
                benchmark_path=harmful_benchmark,
                all_gda_params=cur_gda_params,
                meta_model=cur_meta_model,
                save_folder=save_folder / cur_dataset / "harmful",
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
                    meta_model=cur_meta_model,
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
        save_folder=args.save_folder,
        harmful_benchmarks_folder=args.harmful_benchmarks_folder,
        non_harmful_benchmarks_folder=args.non_harmful_benchmarks_folder,
        num_samples_per_class=args.num_samples_per_class,
        random_state=args.random_state,
        verbose=args.verbose,
    )
