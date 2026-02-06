from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plotting.utils import color_palette, models_names, set_plot_style

set_plot_style()

base_scores = [
    {"model": "llama3_8b_inst", "f1": 0.8625, "num_layers_used": 1, "C": 0.0},
    {"model": "mistral7b_inst", "f1": 0.8418, "num_layers_used": 1, "C": 0.0},
    {"model": "olmo2_1124_7b_inst", "f1": 0.8944, "num_layers_used": 1, "C": 0.0},
    {"model": "qwen3_8b", "f1": 0.8536, "num_layers_used": 1, "C": 0.0},
]

for score in base_scores:
    score["model"] = models_names.get(score["model"], score["model"])
# base_scores = {v["model"]: v["f1"] for v in base_scores}

models_of_interest = [
    "llama3_8b_inst",
    "mistral7b_inst",
    "olmo2_1124_7b_inst",
    "qwen3_8b",
]

if __name__ == "__main__":
    root = Path(__file__).parent.parent / "plotting"
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data" / "different_regularization_data.csv"
    plot_df = pd.read_csv(data_path)
    plot_df = plot_df[plot_df.model.isin(models_of_interest)]
    plot_df.model = plot_df.model.map(models_names)
    # plot_df["C"] = 1 / plot_df["C"]
    plot_df = pd.concat([plot_df, pd.DataFrame(base_scores)])

    plot = sns.lineplot(
        plot_df,
        x="num_layers_used",
        y="f1",
        hue="model",
        palette=color_palette,
    )
    # for model in plot_df.model.unique():
    #     plt.axhline(
    #         base_scores[model],
    #         linestyle="--",
    #         color=color_palette[model],
    #         alpha=0.75,
    #     )
    plt.xlabel("Number of Layers Used")
    plt.ylabel("F1 Score")
    plt.legend(loc="lower right")
    # plt.xlim(left=0.5)
    plt.ylim(bottom=0.82)
    # plt.xscale("log")
    plt.tight_layout()
    plot.get_figure().savefig(
        output_dir / "different_regularization.png",
    )
    plot.get_figure().savefig(
        output_dir / "different_regularization.pdf",
    )
