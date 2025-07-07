from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plotting.commons import FONTSIZE_LABELS, FONTSIZE_LEGEND, FONTSIZE_TICKS
from plotting.utils import set_plot_style

all_datasets_harmful = [
    "Aegis",
    "HarmB",
    "OAI",
    "SimpST",
    "ToxiC",
    "ToxiC hum",
    "WG",
    "WJ",
    "XS",
]
all_datasets_neutral = [
    "Alpaca",
    "BBH",
    "Codex",
    "GSM8k",
    "MMLU",
    "MTBench",
    "TruthfulQA",
]
all_models = [
    "llama_guard_3_8b",
    "wildguard",
    "llama3_8b_inst",
    "mistral7b_inst",
    "olmo2_1124_7b_inst",
]

# The following control what datasets will be plotted
plot_neutral = False
datasets_harmful = [
    "Aegis",
    "HarmB",
    "OAI",
    "SimpST",
    # "ToxiC",
    "ToxiC hum",
    "WG",
    "WJ",
    "XS",
]
datasets_harmful_names = [
    "Aegis",
    "HarmBench",
    "OpenAI",
    "SimpST",
    # "ToxiChat",
    "ToxiChat",
    "WildGuardMix",
    "WildJailbreak",
    "XS",
]
datasets_neutral = [
    "Alpaca",
    "Codex",
    "GSM8k",
    "MMLU",
    "MTBench",
    "TruthfulQA",
]  # Removed BBH
models = [
    "wildguard",
    # "llama3_8b_inst",
    "mistral7b_inst",
    "olmo2_1124_7b_inst",
]
model_names = [
    "WildGuard",
    # "Llama3-8B",
    "Mistral 7B",
    "Olmo2 7B",
]

colormap = {"WGMix": "green", "All w/o WGMix": "red", "All": "orange"}


if __name__ == "__main__":
    set_plot_style()
    root = Path(__file__).parent.parent / "plotting"
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data" / "pretrained_vs_instruct_new.csv"
    df = pd.read_csv(data_path)

    # Loads df with columns: Model    WG  AVG-WG    AVG
    # Create a new flattened df that contains as each row the cell from the original df
    # So that we have something like {"Model": ..., "Setting": [WG / AVG-WG / AVG], "Acc": ...}

    # Create a new dataframe with the columns: Model, Setting, Acc
    data = []
    for i in range(3):
        cur_df = df.iloc[2 * i : 2 * (i + 1)]
        model = cur_df.Model.iloc[0].replace("-PT", "")
        for setting in df.columns[1:3]:
            data.append(
                {
                    "Model": model,
                    "Setting": setting,
                    "Acc-PT": cur_df[setting].iloc[0],
                    "Acc-Inst": cur_df[setting].iloc[1],
                }
            )
    df = pd.DataFrame(data)
    # Add different bar style per model type
    plot = plt.subplot()
    plot = sns.barplot(
        data=df,
        x="Model",
        y="Acc-Inst",
        hue="Setting",
        palette=colormap,
        alpha=0.5,
        dodge=True,
        hatch="",
    )
    plot1 = sns.barplot(
        data=df, x="Model", y="Acc-PT", hue="Setting", palette=colormap, hatch="\\\\"
    )
    plot1.set_ylim(65, 90)
    plot.set_xticklabels(
        plot.get_xticklabels(),
        fontsize=FONTSIZE_TICKS,
    )
    plot.set_xlabel(None)

    plot.set_ylabel("F1 Score", fontsize=FONTSIZE_LABELS)
    plot.set_yticklabels(plot.get_yticklabels(), fontsize=FONTSIZE_TICKS)
    plot.set_ylim(65, 90)

    # Create legend and move it to the bottom right
    # Make it non transparetnt
    _, labels = plot.get_legend_handles_labels()
    labels = labels[2:]
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=colormap[label],
        )
        for label in labels
    ]
    handles.append(
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="\\\\"
        )  # HATCHED white patch
    )
    handles.append(
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="white", edgecolor="black"
        )  # HATCHED white patch
    )
    labels.append("Pretrained")  # Label indicates hatched bars are PT
    labels.append("Instruct")  # Label indicates hatched bars are PT
    legend = plot.legend(
        handles,
        labels,
        title="Evaluation data and model type",
        loc="lower left",
        fontsize=FONTSIZE_LEGEND,
        title_fontsize=FONTSIZE_LEGEND,
        framealpha=1,
        ncol=2,
    )
    plt.tight_layout()
    plot.get_figure().savefig(
        output_dir / "pretrained_vs_instruct.png",
    )
    plot.get_figure().savefig(
        output_dir / "pretrained_vs_instruct.pdf",
    )
