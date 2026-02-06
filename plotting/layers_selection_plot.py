import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from plotting.utils import models_names, set_plot_style

set_plot_style()
# flatten_plots()
# plt.rcParams.update(
#     {
#         "figure.figsize": [8, 4],
#     }
# )

if __name__ == "__main__":
    root = Path(__file__).parent.parent / "plotting"
    output_dir = root / "outputs/layer_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = root / "data" / "layer_selection_data.json"

    with open(data_path, "r") as f:
        data = json.load(f)
    num_layers = data["num_layers"]
    layers_used = data["layers_used"]
    colors = [(1, 1, 1), "gold", "darkred"]  # white, yellow, red
    positions = [0.0, 1e-5, 1.0]  # 0=white, just above 0=yellow, 1=red
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", list(zip(positions, colors)), N=256
    )

    for model_name, layers_dict in layers_used.items():
        all_layers = range(num_layers[model_name])
        c_values = list(layers_dict.keys())
        data = np.zeros((len(c_values), num_layers[model_name]))
        for i, c in enumerate(c_values):
            coeffs = layers_dict[c]
            for j, layer in enumerate(all_layers):
                if j < len(coeffs):
                    data[i, j] = coeffs[j]
        plot = sns.heatmap(
            data,
            xticklabels=list(all_layers),
            yticklabels=[1 / float(c.split("_")[1]) for c in c_values],
            cmap=cmap,
            # cbar_kws={"label": "$|w_i| / \\text{max}(|w|)$"},
            cbar_kws={"label": "Normalized Coefficient"},
            vmin=0,
            vmax=1,
        )
        print_model_name = models_names.get(model_name, model_name)
        plt.yticks(rotation=0)
        plt.xlabel("Layer Number")
        num_skips = 3
        plt.xticks([x+0.5 for x in all_layers[::num_skips]], rotation=0, labels=range(1, num_layers[model_name]+1, num_skips))
        # plt.xticks([x+0.5 for x in all_layers], rotation=45, labels=range(1, num_layers[model_name]+1))
        plt.ylabel("Regularization (C)")
        plt.tight_layout()
        plot.get_figure().savefig(
            output_dir / f"{model_name}.png",
        )
        plot.get_figure().savefig(
            output_dir / f"{model_name}.pdf",
        )
        plt.clf()
