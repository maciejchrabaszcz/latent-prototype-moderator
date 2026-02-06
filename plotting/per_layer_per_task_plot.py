from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plotting.utils import color_palette, flatten_plots, models_names, set_plot_style

set_plot_style()
# flatten_plots()
# models_to_use = [
#     # "llama3_8b_inst",
#     # "mistral12b_inst",
#     "mistral7b_inst",
#     # "olmo2_1124_13b_inst",
#     # "olmo2_1124_7b_inst",
# ]
model_to_use = "mistral7b_inst"
harmful_datasets_of_interest = [
    "aegis",
    # "harmbench",
    # "openai_mod",
    # "simplesafety_tests",
    "toxicchat_humanannotated",
    "wildguardtest",
    # "wildjailbreak_test",
    "xstest",
]

if __name__ == "__main__":
    root = Path(__file__).parent.parent / "plotting"
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data" / "per_layer_per_task_plotting_data.csv"

    plot_df = pd.read_csv(data_path)
    plot_df = plot_df[(plot_df["model"] == model_to_use)  & (plot_df["dataset"].isin(harmful_datasets_of_interest))]

    plot_df["f1"] = plot_df["f1"] * 100
    plot = sns.lineplot(
        plot_df[plot_df.layer != 0],
        x="layer",
        y="f1",
        hue="dataset",
        # palette=color_palette,
    )
    plt.xlabel("Layer")

    # for _, row in majority_df.iterrows():
    #     plt.axhline(
    #         y=row["wildguardtest"], linestyle="--", color=color_palette[row["model"]]
    #     )
    plt.ylabel("F1 Score")
    plt.legend(title=None)
    plt.tight_layout()
    plot.get_figure().savefig(
        output_dir / "per_layer_per_task_performance.png",
    )
    plot.get_figure().savefig(
        output_dir / "per_layer_per_task_performance.pdf",
    )
