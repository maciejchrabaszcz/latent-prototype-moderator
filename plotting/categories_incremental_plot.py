from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plotting.utils import color_palette, flatten_plots, models_names, set_plot_style

set_plot_style()
flatten_plots()

models_to_use = [
    "llama3_8b_inst",
    # "mistral12b_inst",
    "mistral7b_inst",
    # "olmo2_1124_13b_inst",
    "olmo2_1124_7b_inst",
]


model_scores = {
    "Llama3-8B": 0.8625,
    "Mistral-7B": 0.8418,
    "OLMo2-7B": 0.8944,
}

if __name__ == "__main__":
    root = Path(__file__).parent.parent / "plotting"
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data" / "wgmix_categories_results.csv"
    df = pd.read_csv(data_path)
    plot_df = df[
        (df["model_type"] == "Hidden States GDA")
        & (df["dataset_type"] == "harmful")
        & (df["model"].isin(models_to_use))
    ]

    plot_df["model"] = plot_df["model"].map(models_names)
    plot_df["score"] = plot_df["score"] * 100
    plot = sns.lineplot(
        data=plot_df,
        x="num_categories",
        y="score",
        hue="model",
        marker="o",
        palette=color_palette,
    )

    # for model, score in model_scores.items():
    #     plt.axhline(y=score, linestyle="--", color=color_palette[model])
    plt.xlabel("Number of risk categories")
    plt.ylabel("F1 Score on WGMix")
    plt.legend(title=None)
    plt.tight_layout()
    plot.get_figure().savefig(
        output_dir / "risk_categories.png",
    )
    plot.get_figure().savefig(
        output_dir / "risk_categories.pdf",
    )
