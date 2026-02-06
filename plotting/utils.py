import matplotlib.pyplot as plt


def set_plot_style():
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size' : 15,                   # Set font size to 11pt
        'axes.labelsize': 15,               # -> axis labels
        'xtick.labelsize':12,
        'ytick.labelsize':12,
        'legend.fontsize': 12,
        'lines.linewidth':2,
        'text.usetex': False,
        'pgf.rcfonts': False,
    })

def flatten_plots():
    plt.rcParams.update({
        "figure.figsize": [6.4, 3.2],
    })

models_names = {
    "llama3_8b_inst": "Llama3-8B",
    "mistral12b_inst": "Mistral-12B",
    "mistral7b_inst": "Mistral-7B",
    "olmo2_1124_7b_inst": "OLMo2-7B",
    "olmo2_1124_13b_inst": "OLMo2-13B",
    "qwen3_8b": "Qwen3-8B",
}

color_palette = {
    "Llama3-8B": "#0568e2",
    "Mistral-7B": "#ff7001",
    "OLMo2-7B": "#f0529c",
    "Qwen3-8B": "#8b33ff",
}