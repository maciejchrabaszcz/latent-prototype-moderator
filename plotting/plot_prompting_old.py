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


if __name__ == "__main__":
    root = Path(__file__).parent.parent / "plotting"
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data" / "prompting_data.csv"
    df = pd.read_csv(data_path)

    # Filter out unwanted models
    df = df[df["model"].isin(models)]
    # Rename models to target tags
    model_name_dict = {
        org_name: new_name for org_name, new_name in zip(models, model_names)
    }
    df["model"] = df["model"].replace(model_name_dict)

    # Get rid of datasets we don't want
    df = df[["model"] + datasets_harmful + datasets_neutral]

    df["Harmful"] = df[datasets_harmful].mean(axis=1)
    df["Neutral"] = df[datasets_neutral].mean(axis=1)

    if plot_neutral:
        df = df[["model"] + datasets_harmful + ["Neutral"]]
        # Rename harmful col names
        df.columns = ["Model"] + datasets_harmful_names + ["Neutral"]
    else:
        # Rename harmful col names
        df = df[["model"] + datasets_harmful]
        df.columns = ["Model"] + datasets_harmful_names

    org_df = df

    # Parse dataframe into the dict format, where each entry contains accuracy of the given model on a specific dataset
    data = []
    for idx, row in df.iterrows():
        model = row["Model"]
        for dataset, acc in zip(df.columns[1:], row[1:]):
            data.append({"Model": model, "Dataset": dataset, "Acc": 100 * acc})
    df = pd.DataFrame(data)

    # Create barplot with acc per dataset for each model
    plot = sns.barplot(
        data=df,
        x="Dataset",
        y="Acc",
        hue="Model",
    )

    # Set tick and label fontisizes
    plot.set_xticklabels(
        plot.get_xticklabels(),
        rotation=30,
        horizontalalignment="right",
        fontsize=FONTSIZE_TICKS,
    )
    plot.set_yticklabels(plot.get_yticklabels(), fontsize=FONTSIZE_TICKS)
    plot.set_ylabel("Accuracy", fontsize=FONTSIZE_LABELS)
    plot.set_xlabel(None)

    # Create legend and move it to the bottom right
    handles, labels = plot.get_legend_handles_labels()
    legend = plot.legend(
        handles,
        labels,
        title="Model",
        loc="lower right",
        fontsize=FONTSIZE_LEGEND,
        title_fontsize=FONTSIZE_LEGEND,
    )

    plt.tight_layout()
    # Save png
    output_path = output_dir / "prompting.png"
    plot.get_figure().savefig(
        output_path,
    )
    # Save pdf version
    plot.get_figure().savefig(
        output_path.with_suffix(".pdf"),
    )
    print("Saved plot to ", output_path)
