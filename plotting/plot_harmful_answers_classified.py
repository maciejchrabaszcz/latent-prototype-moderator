from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from plotting.commons import FONTSIZE_LEGEND, FONTSIZE_TICKS

models = ["llama3_8b_inst", "mistral7b_inst", "olmo2_1124_7b_inst"]
model_names = ["Llama3-8B", "Mistral-7B", "OLMo2-7B"]

metrics = [
    "num_harfmul_questions",
    "num_properly_classified",
    "num_harmful_responses",
    "num_is_harmful_and_was_predicted_as_harmful",
]
NUM_Q = "Num questions"
NUM_DETECTED = "Correctly detected harmful inputs"
NUM_HARMFUL_RESPONSES = "Model produces harmful output"
NUM_HARMFUL_IDENTIFIED = "Model identifies malicious input"
# NUM_HARMFUL_IDENTIFIED = "Model identifies malicious input but answers it"
metric_names = [
    NUM_Q,
    NUM_DETECTED,
    NUM_HARMFUL_RESPONSES,
    NUM_HARMFUL_IDENTIFIED,
]
valid_metric_names = [
    # NUM_Q,
    NUM_HARMFUL_RESPONSES,
    NUM_HARMFUL_IDENTIFIED,
]
color_map = {
    NUM_Q: "tab:green",
    NUM_HARMFUL_IDENTIFIED: "tab:green",
    NUM_HARMFUL_RESPONSES: "tab:red",
}

if __name__ == "__main__":
    root = Path(__file__).parent.parent / "plotting"
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data" / "harmful_responses_detected.csv"
    df = pd.read_csv(data_path)

    model_name_dict = {
        org_name: new_name for org_name, new_name in zip(models, model_names)
    }
    df["model"] = df["model"].replace(model_name_dict)
    org_df = df

    print(list(df.columns[1:]))

    # Parse dataframe into the dict format, where each entry contains accuracy of the given model on a specific dataset
    data = []
    for idx, row in df.iterrows():
        model = row["model"]
        for metric, val in zip(df.columns[1:], row[1:]):
            data.append({"Model": model, "Metric": metric, "Val": val})
    df = pd.DataFrame(data)
    # Rename metric columns
    metric_name_dict = {
        org_name: new_name for org_name, new_name in zip(metrics, metric_names)
    }
    df["Metric"] = df["Metric"].replace(metric_name_dict)
    # Drop total number of questions
    df = df[df["Metric"].isin(valid_metric_names)]

    # Sort the metrics by their order in colormap
    df["Metric"] = pd.Categorical(
        df["Metric"], categories=valid_metric_names, ordered=True
    )

    # Create barplot with acc per dataset for each model
    plot = sns.barplot(
        data=df,
        x="Model",
        y="Val",
        hue="Metric",
        palette=color_map,
    )

    # Set tick and label fontisizes
    plot.set_xticklabels(
        plot.get_xticklabels(),
        fontsize=FONTSIZE_TICKS,
    )
    plot.set_yticklabels(plot.get_yticklabels(), fontsize=FONTSIZE_TICKS)
    plot.set_ylabel(None)
    plot.set_xlabel(None)
    plot.set_yscale("log")
    plt.minorticks_off()
    plot.set_ylim(2, 2000)
    # Set ticks only for 1, 3, 10, 30, etc
    plot.set_yticks([10, 100, 1000])
    plot.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    # Create legend and move it to the bottom right
    handles, labels = plot.get_legend_handles_labels()
    legend = plot.legend(
        handles,
        labels,
        title=None,
        loc="lower center",
        fontsize=FONTSIZE_LEGEND,
        # Make it nontransparent
        framealpha=1,
    )

    plt.tight_layout()
    # Save png
    output_path = output_dir / "harmful_responses_detected.png"
    plot.get_figure().savefig(
        output_path,
    )
    # Save pdf version
    plot.get_figure().savefig(
        output_path.with_suffix(".pdf"),
    )
    print("Saved plot to ", output_path)
