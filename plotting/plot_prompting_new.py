from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from plotting.commons import FONTSIZE_LABELS, FONTSIZE_LEGEND, FONTSIZE_TICKS
from plotting.utils import set_plot_style
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
    "Harm",
    "OAIMod",
    "SimpST",
    "ToxiC",
    "WG",
    "WJB",
    "XS",
]
datasets_neutral = [
    "Alpaca",
    "BBH",
    "Codex",
    "GSM8k",
    "MMLU",
    "MTBench",
    "TruthfulQA",
]
models = [
    "llama3_8b_inst",
    "mistral7b_inst",
    "olmo2_1124_7b_inst",
]
guards = [
    "wildguard",
    "llama_guard_3_8b",
]
label_to_name = {
    "llama3_8b_inst": "Llama3-8B",
    "mistral7b_inst": "Mistral-7B",
    "olmo2_1124_7b_inst": "OLMo2-7B",
    "llama_guard_3_8b": "LlamaGuard",
    "wildguard": "WildGuard",
}
color_palette = {
    "llama3_8b_inst": "#0568e2",
    "mistral7b_inst": "#ff7001",
    "olmo2_1124_7b_inst": "#f0529c",
    "llama_guard_3_8b": "#e6e861",
    "wildguard": "#a5dcaf",
}

if __name__ == "__main__":
    set_plot_style()
    root = Path(__file__).parent.parent / "plotting"
    output_dir = root / "outputs"
    print(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data" / "prompting_raw_data.csv"
    df = pd.read_csv(data_path, header=[0, 1])
    df.columns = ["Model", "Prompt ID"] + datasets_harmful + datasets_neutral

    # Ignore neutral datasets
    df = df[["Model", "Prompt ID"] + datasets_harmful]

    aggregate_scores = []
    # Add rows for max, mean and min values for each dataset for mistral / olmo / llama
    for model in models:
        all_prompt_scores = df[df["Model"] == model][datasets_harmful].values
        min_score_per_dataset = all_prompt_scores.min(axis=0)
        max_score_per_dataset = all_prompt_scores.max(axis=0)
        mean_score_per_dataset = all_prompt_scores.mean(axis=0)
        # Add three rows for the model with prompt id for min, max and mean
        min_row = {
            "Model": model,
            "Prompt ID": "Min",
            **{
                dataset: min_score
                for dataset, min_score in zip(datasets_harmful, min_score_per_dataset)
            },
        }
        max_row = {
            "Model": model,
            "Prompt ID": "Max",
            **{
                dataset: max_score
                for dataset, max_score in zip(datasets_harmful, max_score_per_dataset)
            },
        }
        mean_row = {
            "Model": model,
            "Prompt ID": "Mean",
            **{
                dataset: mean_score
                for dataset, mean_score in zip(datasets_harmful, mean_score_per_dataset)
            },
        }
        aggregate_scores.extend([min_row, max_row, mean_row])

    # Concatenate guard scores to the dataframe with aggregate scores
    for guard in guards:
        guard_score = df[df["Model"] == guard][datasets_harmful].values[0]
        aggregate_scores.append(
            {
                "Model": guard,
                "Prompt ID": None,
                **{
                    dataset: score
                    for dataset, score in zip(datasets_harmful, guard_score)
                },
            }
        )
    df = pd.DataFrame(aggregate_scores)
    # Make Avg column use bold font in latex
    df["Avg"] = df[datasets_harmful].mean(axis=1)

    plt.figure(figsize=(10, 5))

    # Create a bar plot, where for a guard model we plot the scores for each dataset,
    # while for the other models we plot mean score as the bar and min and max as error bars
    # Parse the dataframe into the format required
    data = []
    for idx, row in df.iterrows():
        model = row["Model"]
        prompt_id = row["Prompt ID"]
        for dataset, acc in zip(df.columns[2:], row[2:]):
            data.append(
                {
                    "Model": model,
                    "Prompt ID": prompt_id,
                    "Dataset": dataset,
                    "Acc": 100 * acc,
                }
            )
    df = pd.DataFrame(data)

    # Create barplot with acc per dataset for each model
    # Sample means from the df and the part of the df for guards
    # mean_df = df[df["Prompt ID"] == "Mean"]
    # guard_df = df[df["Model"].isin(guards)]
    # plot_df = pd.concat([mean_df, guard_df], ignore_index=True)
    plot = sns.barplot(
        data=df, x="Dataset", y="Acc", hue="Model", palette=color_palette, capsize=0.6
    )

    plot.set_xticklabels(
        plot.get_xticklabels(),
        rotation=30,
        horizontalalignment="right",
        fontsize=FONTSIZE_TICKS + 2,
    )
    plot.set_yticklabels(plot.get_yticklabels(), fontsize=FONTSIZE_TICKS + 2)
    plot.set_ylabel("F1 score", fontsize=FONTSIZE_LABELS + 2)
    plot.set_xlabel(None)

    # Create legend and move it to the bottom right, make it 5 cols, make it non-transparent
    handles, labels = plot.get_legend_handles_labels()
    labels = [label_to_name[label] for label in labels]
    legend = plot.legend(
        handles,
        labels,
        title=None,
        loc="lower right",
        fontsize=FONTSIZE_LEGEND,
        ncols=3,
        framealpha=1,
    )

    plt.tight_layout()
    plot.get_figure().savefig(
        output_dir / "prompting_aggregate.png",
    )
    plot.get_figure().savefig(
        output_dir / "prompting_aggregate.pdf",
    )
