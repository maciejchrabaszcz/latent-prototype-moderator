from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from plotting.commons import FONTSIZE_LABELS, FONTSIZE_LEGEND, FONTSIZE_TICKS

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

colormap = {"WGMix": "green", "All-WGMix": "red", "All": "orange"}


if __name__ == "__main__":
    root = Path(__file__).parent.parent / "plotting"
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data" / "pretrained_vs_instruct.csv"
    df = pd.read_csv(data_path)

    # Loads df with columns: Model    WG  AVG-WG    AVG
    # Create a new flattened df that contains as each row the cell from the original df
    # So that we have something like {"Model": ..., "Setting": [WG / AVG-WG / AVG], "Acc": ...}

    # Create a new dataframe with the columns: Model, Setting, Acc
    data = []
    for idx, row in df.iterrows():
        model = row["Model"]
        for setting, acc in zip(df.columns[1:], row[1:]):
            data.append({"Model": model, "Setting": setting, "Acc": acc})
    df = pd.DataFrame(data)

    print(df["Setting"].unique().tolist())

    plot = sns.barplot(data=df, x="Model", y="Acc", hue="Setting", palette=colormap)

    plot.set_xticklabels(
        plot.get_xticklabels(),
        fontsize=FONTSIZE_TICKS,
    )
    plot.set_xlabel(None)

    plot.set_ylabel("F1 difference", fontsize=FONTSIZE_LABELS)
    plot.set_yticklabels(plot.get_yticklabels(), fontsize=FONTSIZE_TICKS)
    plot.set_ylim(0, 15)

    # Create legend and move it to the bottom right
    # Make it non transparetnt
    handles, labels = plot.get_legend_handles_labels()
    legend = plot.legend(
        handles,
        labels,
        title="Evaluation data",
        loc="upper left",
        fontsize=FONTSIZE_LEGEND,
        title_fontsize=FONTSIZE_LEGEND,
        framealpha=1,
        ncol=2,
    )

    plot.get_figure().savefig(
        output_dir / "pretrained_vs_instruct.png",
    )
    plot.get_figure().savefig(
        output_dir / "pretrained_vs_instruct.pdf",
    )
