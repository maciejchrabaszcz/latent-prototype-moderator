from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plotting.utils import color_palette, flatten_plots, models_names, set_plot_style

set_plot_style()
flatten_plots()

datasets_of_interest = [
    "aegis",
    "toxicchat_humanannotated",
    "wildguardtest",
]

combinations_of_interest = [
    "wildguard",
    "wildguard+aegis",
    "wildguard+toxichat",
    "wildguard+aegis+toxichat",
]

names_of_combinations = {
    "wildguard": "WG",
    "wildguard+aegis": "WG + A",
    "wildguard+toxichat": "WG + TC",
    "wildguard+aegis+toxichat": "WG + A + TC",
}


if __name__ == "__main__":
    root = Path(__file__).parent.parent / "plotting"
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data" / "multiple_datasets_plot_data.csv"
    df = pd.read_csv(data_path)
    df = df[
        df["dataset"].isin(datasets_of_interest)
        & (df["model_type"] == "Hidden States GDA")
        & (df["num_datasets"].isin(combinations_of_interest))
        & (df["dataset_type"] == "harmful")
    ]
    df["num_datasets"] = df["num_datasets"].map(names_of_combinations)
    df["model"] = df["model"].map(models_names)

    df = (
        df.groupby(["num_datasets", "model", "dataset_type"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            std_score=("score", "std"),
        )
        .reset_index(drop=True)
    )

    df["num_datasets"] = pd.Categorical(
        df["num_datasets"],
        categories=[
            "WG",
            "WG + A",
            "WG + TC",
            "WG + A + TC",
        ],
        ordered=True,
    )
    df["mean_score"] = df["mean_score"] * 100
    plot = sns.lineplot(
        data=df,
        x="num_datasets",
        y="mean_score",
        hue="model",
        # style="dataset",
        markers=True,
        dashes=False,
        # disable confidence intervals
        errorbar=None,
        marker="o",
        palette=color_palette,
    )
    plt.xlabel("Datasets used for prototypes")
    plt.ylabel("Mean F1 Score")
    plt.legend(title=None, loc="lower right")
    plt.tight_layout()
    plot.get_figure().savefig(
        output_dir / "dataset_incremental.png",
    )
    plot.get_figure().savefig(
        output_dir / "dataset_incremental.pdf",
    )
