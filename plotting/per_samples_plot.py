from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
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

model_num_layers = {
    "llama3_8b_inst": 32,
    "mistral7b_inst": 32,
    "olmo2_1124_7b_inst": 32,
    "olmo2_1124_13b_inst": 40,
    "mistral12b_inst": 40,
    "olmo2_0325_32b_inst": 60,
}


if __name__ == "__main__":
    root = Path(__file__).parent.parent / "plotting"
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data" / "per_samples_data.csv"

    plot_df = pd.read_csv(data_path)
    plot_df = plot_df[plot_df["model"].isin(models_to_use)]
    plot_df.model = plot_df.model.map(models_names)
    plot_df["f1"] = plot_df["f1"] * 100
    fig, ax = plt.subplots()  # Use fig, ax for clarity
    plot = sns.lineplot(
        data=plot_df,  # Pass data explicitly
        x="num_samples",
        y="f1",
        hue="model",
        alpha=0.75,
        style="model",
        markers=True,
        dashes=False,
        palette=color_palette,
        ax=ax,  # Pass the ax object to seaborn
    )

    ax.set_xscale("log")
    # Set major ticks and their formatter
    ax.set_xticks([10, 100, 1000, 10000, 40000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.minorticks_off()
    plt.xlabel("Num Samples Per Class")
    plt.ylabel("F1 Score on WGMix")
    plt.legend(title=None)
    plt.tight_layout()
    plot.get_figure().savefig(
        output_dir / "per_samples_performance.png",
    )
    plot.get_figure().savefig(
        output_dir / "per_samples_performance.pdf",
    )
