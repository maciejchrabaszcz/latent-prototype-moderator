from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
from plotting.commons import FONTSIZE_LABELS, FONTSIZE_LEGEND, FONTSIZE_TICKS
from plotting.utils import set_plot_style

color_palette = {
    "OLMo2": "#f0529c",
    "OLMoE": "#f0529c",
    "Mistral": "#ff7001",
    "Llama3": "#0568e2",
    "Qwen3": "#8b33ff",
    "Qwen3-MoE": "#8b33ff",
    "WildGuard": "#a5dcaf",
    "Aegis-D": "#beae8a",
    "LlamaGuard3": "#e6e861",
}
MARKER_DICT = {
    "OLMo2": "o",
    "OLMoE": "*",
    "Mistral": "o",
    "Llama3": "o",
    "Qwen3": "o",
    "Qwen3-MoE": "*",
    "WildGuard": "P",
    "Aegis-D": "P",
    "LlamaGuard3": "P",
}
MARKERSIZE_LPM = 80
MARKERSIZE_LPM_MOE = 150
MARKERSIZE_GUARD = 150
MARKER_SIZE_DICT = {
    "OLMo2": MARKERSIZE_LPM,
    "OLMoE": MARKERSIZE_LPM_MOE,
    "Mistral": MARKERSIZE_LPM,
    "Llama3": MARKERSIZE_LPM,
    "Qwen3": MARKERSIZE_LPM,
    "Qwen3-MoE": MARKERSIZE_LPM_MOE,
    "WildGuard": MARKERSIZE_GUARD,
    "Aegis-D": MARKERSIZE_GUARD,
    "LlamaGuard3": MARKERSIZE_GUARD,
}

# LEGEND_LABELS_DICT = {
#     "Aegis-D": "Aegis-D",
#     "LlamaGuard3": "LlamaGuard3",
#     "WildGuard": "WildGuard",
#     "Mistral": "LPM(Mistral)",
#     "Llama3": "LPM(Llama3)",
#     "OLMo2": "LPM(OLMo2)",
#     "OLMoE": "LPM(OLMoE)",
#     "Qwen3": "LPM(Qwen3)",
#     "Qwen3-MoE": "LPM(Qwen3-MoE)",
# }
LEGEND_LABELS_DICT = {
    "Aegis-D": "Aegis-D",
    "LlamaGuard3": "LlamaGuard3",
    "WildGuard": "WildGuard",
    "Mistral": "Mistral",
    "Llama3": "Llama3",
    "OLMo2": "OLMo2",
    "OLMoE": "OLMoE",
    "Qwen3": "Qwen3",
    "Qwen3-MoE": "Qwen3-MoE",
}
LINEPLOT_MODELS = ["Mistral", "OLMo2", "Llama3", "Qwen3"]

if __name__ == "__main__":
    set_plot_style()
    plt.rcParams.update({
        "figure.figsize": [8.4, 4.2],
    })
    root = Path(__file__).parent.parent / "plotting"
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data" / "model_size_perf.csv"
    df = pd.read_csv(data_path)

    lineplot_df = df[df["Model Type"].isin(LINEPLOT_MODELS)]
    plot = sns.lineplot(
        lineplot_df,
        x="Active params (B)",
        y="Harm Acc",
        hue="Model Type",
        palette=color_palette,
    )

    # Draw points
    sns.scatterplot(
        df,
        x="Active params (B)",
        y="Harm Acc",
        hue="Model Type",
        style="Model Type",
        palette=color_palette,
        markers=MARKER_DICT,
        size="Model Type",
        sizes=MARKER_SIZE_DICT,
    )

    # Set xticks to selected values
    # plt.xscale("log")
    # plot.set_xticks([1, 7, 13, 32, 70])
    # plot.set_xticklabels(
    #     ["1B", "7B", "13B", "32B", "70B"], fontsize=FONTSIZE_TICKS
    # )
    plot.set_xlabel("Active params", fontsize=FONTSIZE_LABELS)

    # Set ytick sizes
    plot.set_yticklabels(plot.get_yticklabels(), fontsize=FONTSIZE_TICKS)
    plot.set_ylabel("Harmful Detection F1", fontsize=FONTSIZE_LABELS)
    plt.xscale("log")
    plot.set_xticks([1, 7, 32, 70])
    plot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.minorticks_off()
    plot.set_xticklabels(
        ["1B", "7B", "32B", "70B"], fontsize=FONTSIZE_TICKS
    )
    # Replot legend
    # Get handles and labels from the plot
    handles, labels = plot.get_legend_handles_labels()

    # Remove first 2 handles and labels - the lineplot artifcats
    handles = handles[len(LINEPLOT_MODELS) :]
    labels = labels[len(LINEPLOT_MODELS) :]

    # Reorder the labels according to the legend_labels_dict and rename
    new_handles, new_labels = [], []
    for label in LEGEND_LABELS_DICT.keys():
        if label in labels:
            idx = labels.index(label)
            new_handles.append(handles[idx])
            new_labels.append(LEGEND_LABELS_DICT[label])

    # We have now 7 handles and labels, with 3 corresponding to guards and 4 to our method
    # Plot the guards as the first column, and our method as the second column in the legend
    invisible_handle = Line2D([], [], color="none", label="Guards")
    new_handles.insert(0, invisible_handle)  # Insert at the desired position
    new_labels.insert(0, "Guards")
    invisible_handle = Line2D([], [], color="none", label="LPM")
    new_handles.insert(4, invisible_handle)  # Insert at the desired position
    new_labels.insert(4, "LPM")  # Add an empty label for spacing
    # drop OLMOE and Qwen3-MoE from the legend
    new_handles = [h for h, l in zip(new_handles, new_labels) if l not in ["OLMoE", "Qwen3-MoE"]]
    new_labels = [l for l in new_labels if l not in ["OLMoE", "Qwen3-MoE"]]

    moe_handle = Line2D([], [], color="black", marker="*", linestyle="None", label="MoE")
    new_handles.append(moe_handle)
    new_labels.append("MoE")

    # Create new legend
    legend = plot.legend(
        new_handles,
        new_labels,
        title=None,
        loc="center left",         # <-- Change this
        bbox_to_anchor=(1, 0.5), # <-- Add this
        fontsize=FONTSIZE_LEGEND,
        ncol=1,
    )
    plt.tight_layout()
    plot.get_figure().savefig(
        output_dir / "model_size_perf.png",
    )
    plot.get_figure().savefig(
        output_dir / "model_size_perf.pdf",
    )
