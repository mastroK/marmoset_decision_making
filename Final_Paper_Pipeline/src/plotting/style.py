"""Shared matplotlib style setup, from config.yaml's `plotting` section."""

import matplotlib.pyplot as plt


def apply_style(config):
    p = config["plotting"]
    plt.rcParams["font.family"] = p["font_family"]
    plt.rcParams["font.size"] = p["fontsize"]
    plt.rcParams["axes.linewidth"] = p["line_width"]
    plt.rcParams["xtick.major.width"] = p["line_width"]
    plt.rcParams["ytick.major.width"] = p["line_width"]
    plt.rcParams["xtick.major.size"] = p["tick_length"]
    plt.rcParams["ytick.major.size"] = p["tick_length"]
    plt.rcParams["lines.linewidth"] = p["line_width"]
    plt.rcParams["svg.fonttype"] = p["svg_fonttype"]


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
