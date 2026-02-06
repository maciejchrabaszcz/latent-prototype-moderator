from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plotting.utils import color_palette, flatten_plots, models_names, set_plot_style

set_plot_style()
# flatten_plots()
models_to_use = [
    "llama3_8b_inst",
    # "mistral12b_inst",
    "mistral7b_inst",
    # "olmo2_1124_13b_inst",
    "olmo2_1124_7b_inst",
    "qwen3_8b"
]

model_num_layers = {
    "llama3_8b_inst": 32,
    "mistral7b_inst": 32,
    "olmo2_1124_7b_inst": 32,
    "olmo2_1124_13b_inst": 40,
    "mistral12b_inst": 40,
    "olmo2_0325_32b_inst": 60,
    "qwen3_8b": 36,
}


if __name__ == "__main__":
    root = Path(__file__).parent.parent / "plotting"
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data" / "per_layer_plotting_data.csv"
    majority_data_path = root / "data" / "majority_per_layer.csv"

    plot_df = pd.read_csv(data_path)
    plot_df = plot_df[plot_df["model"].isin(models_to_use)]
    plot_df["normalized_layer"] = plot_df.apply(
        lambda x: x.layer / model_num_layers[x.model], axis=1
    )
    plot_df.model = plot_df.model.map(models_names)

    majority_df = pd.read_csv(majority_data_path)
    majority_df.model = majority_df.model.str.replace("Majority Vote ", "")
    majority_df = majority_df[majority_df["model"].isin(models_to_use)]
    majority_df.model = majority_df.model.map(models_names)
    plot_df["f1"] = plot_df["f1"] * 100
    plot = sns.lineplot(
        plot_df[plot_df.layer != 0],
        x="layer",
        y="f1",
        hue="model",
        palette=color_palette,
    )
    plt.xlabel("Layer")

    # for _, row in majority_df.iterrows():
    #     plt.axhline(
    #         y=row["wildguardtest"], linestyle="--", color=color_palette[row["model"]]
    #     )
    plt.ylabel("F1 Score on WGMix")
    plt.legend(title=None)
    plt.tight_layout()
    plot.get_figure().savefig(
        output_dir / "per_layer_performance.png",
    )
    plot.get_figure().savefig(
        output_dir / "per_layer_performance.pdf",
    )
