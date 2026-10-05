"""Generates Figures 2-4: plots of the numerical results computed by
experiment.py (results_illustration.json). This script only visualizes
those numbers -- it does not compute anything new.

Colors are the validated categorical palette (blue/green/orange/red),
each series also distinguished by marker and line style so the figures
remain legible in grayscale print.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
GREEN = "#008300"
ORANGE = "#eb6834"
RED = "#e34948"

with open("../results_illustration.json") as f:
    results = json.load(f)


def fig_runtime_scaling():
    rows = results["runtime"]
    p_max = [r["p_max"] for r in rows]
    exact_ms = [r["exact_ms"] for r in rows]
    fptas_ms = [r["fptas_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(p_max, exact_ms, color=BLUE, marker="o", markersize=6,
            linewidth=2, linestyle="-", label="Exact DP (Theorem 3)")
    ax.plot(p_max, fptas_ms, color=ORANGE, marker="s", markersize=6,
            linewidth=2, linestyle="--", label="FPTAS (Theorem 4)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$p_{\max}$ (processing-time range)")
    ax.set_ylabel("mean wall-clock time (ms)")
    ax.set_title("Exact DP vs. FPTAS runtime, $n=40$, $\\epsilon=0.1$,\nweights up to $10^6$ (scaling active)")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", axis="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig("runtime_scaling.pdf", bbox_inches="tight")
    fig.savefig("runtime_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_accuracy_vs_n():
    acc = results["accuracy"]
    ns = sorted(int(k) for k in acc.keys())
    series = [
        ("fptas01", r"FPTAS ($\epsilon=0.1$)", BLUE, "o", "-"),
        ("repair", "Greedy repair (M--H style)", ORANGE, "s", "--"),
        ("edd_skip", "EDD with skipping", GREEN, "^", "-."),
        ("naive", "Naive EDD (no repair)", RED, "D", ":"),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    for key, label, color, marker, ls in series:
        ax.plot(ns, [acc[str(n)][key] for n in ns], color=color,
                marker=marker, markersize=5, linewidth=2, linestyle=ls,
                label=label)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200, 400])
    ax.set_xticklabels(["10", "20", "50", "100", "200", "400"])
    ax.set_xlabel("$n$ (number of sites)")
    ax.set_ylabel("mean ratio to true optimum")
    ax.set_ylim(0.4, 1.04)
    ax.set_title("Accuracy across instance sizes (weights $\\leq 100$)")
    ax.legend(frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.95, loc="center right", bbox_to_anchor=(1.0, 0.68), fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig("accuracy_vs_n.pdf", bbox_inches="tight")
    fig.savefig("accuracy_vs_n.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_epsilon_sensitivity():
    rows = results["scaling_active"]
    eps = sorted({r["eps"] for r in rows})

    def pick(regime, n, key):
        return [next(r[key] for r in rows if r["regime"] == regime
                     and r["n"] == n and r["eps"] == e) for e in eps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.5))
    ax1.plot(eps, [1 - e for e in eps], color="black", linestyle=":",
             linewidth=1.6, label=r"guarantee $1-\epsilon$")
    ax1.plot(eps, pick("adversarial (sub-K sites)", 100, "mean_ratio"),
             color=RED, marker="D", markersize=5, linewidth=2,
             label="adversarial family")
    ax1.plot(eps, pick("adversarial (sub-K sites)", 100, "mean_ratio_completed"),
             color=GREEN, marker="^", markersize=5, linewidth=2,
             linestyle="-.", label="adversarial, with completion")
    ax1.plot(eps, pick("uniform weights <= 10^6", 100, "min_ratio"),
             color=BLUE, marker="o", markersize=5, linewidth=2,
             linestyle="--", label=r"random, $w\leq10^6$ (worst of 30)")
    ax1.set_xlabel(r"$\epsilon$")
    ax1.set_ylabel("ratio to optimum")
    ax1.set_ylim(0.4, 1.03)
    ax1.invert_xaxis()
    ax1.set_title("Accuracy ($n=100$)")
    ax1.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, axis="y", alpha=0.25)

    cells = pick("uniform weights <= 10^6", 100, "mean_cells")
    ms = pick("uniform weights <= 10^6", 100, "mean_ms")
    ax2.plot(eps, [c / 1e6 for c in cells], color=ORANGE, marker="s",
             markersize=6, linewidth=2, linestyle="--", label="table cells ($10^6$)")
    ax2.set_xlabel(r"$\epsilon$")
    ax2.set_ylabel("DP table cells ($\\times10^6$)", color=ORANGE)
    ax2.invert_xaxis()
    ax3 = ax2.twinx()
    ax3.plot(eps, ms, color=BLUE, marker="o", markersize=5, linewidth=2,
             label="time (ms)")
    ax3.set_ylabel("mean time (ms)", color=BLUE)
    ax2.set_title("Cost ($n=100$, random $w\\leq10^6$)")
    for ax in (ax2, ax3):
        ax.spines["top"].set_visible(False)
    ax2.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig("epsilon_sensitivity.pdf", bbox_inches="tight")
    fig.savefig("epsilon_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_runtime_scaling()
    fig_accuracy_vs_n()
    fig_epsilon_sensitivity()
    print("wrote runtime_scaling.pdf, accuracy_vs_n.pdf, epsilon_sensitivity.pdf (+ .png previews)")
