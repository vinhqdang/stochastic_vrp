"""Generates Figures 2-4: plots of the numerical results already
computed and verified by experiment.py (results_illustration.json).
This script only visualizes existing, verified numbers -- it does not
compute anything new.

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
    ax.set_title("Exact DP vs. FPTAS runtime, $n=40$, $\\epsilon=0.1$")
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
    fptas02 = [acc[str(n)]["fptas02_ratio"] for n in ns]
    fptas01 = [acc[str(n)]["fptas01_ratio"] for n in ns]
    repair = [acc[str(n)]["repair_ratio"] for n in ns]
    naive = [acc[str(n)]["naive_ratio"] for n in ns]

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(ns, fptas02, color=BLUE, marker="o", markersize=5,
            linewidth=2, linestyle="-", label=r"FPTAS ($\epsilon=0.2$)")
    ax.plot(ns, fptas01, color=GREEN, marker="^", markersize=5,
            linewidth=2, linestyle="-.", label=r"FPTAS ($\epsilon=0.1$)")
    ax.plot(ns, repair, color=ORANGE, marker="s", markersize=5,
            linewidth=2, linestyle="--", label="Greedy repair (M--H style)")
    ax.plot(ns, naive, color=RED, marker="D", markersize=5,
            linewidth=2, linestyle=":", label="Naive EDD (no repair)")
    ax.set_xlabel("$n$ (number of sites)")
    ax.set_ylabel("mean ratio to true optimum")
    ax.set_ylim(0, 1.08)
    ax.set_title("Accuracy across instance sizes")
    ax.legend(frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.95, loc="upper right", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig("accuracy_vs_n.pdf", bbox_inches="tight")
    fig.savefig("accuracy_vs_n.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_epsilon_sensitivity():
    rows = results["epsilon_sensitivity"]
    eps = [r["eps"] for r in rows]
    ratio = [r["mean_ratio"] for r in rows]
    time_ms = [r["mean_ms"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4))

    ax1.plot(eps, ratio, color=BLUE, marker="o", markersize=6,
             linewidth=2, linestyle="-")
    ax1.set_xlabel(r"$\epsilon$")
    ax1.set_ylabel("mean ratio to optimum")
    ax1.set_ylim(0.995, 1.001)
    ax1.invert_xaxis()
    ax1.set_title("Accuracy")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, axis="y", alpha=0.25)

    ax2.plot(eps, time_ms, color=ORANGE, marker="s", markersize=6,
             linewidth=2, linestyle="--")
    ax2.set_xlabel(r"$\epsilon$")
    ax2.set_ylabel("mean time (ms)")
    ax2.invert_xaxis()
    ax2.set_title("Runtime")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle(r"FPTAS sensitivity to $\epsilon$, $n=30$", y=1.03)
    fig.tight_layout()
    fig.savefig("epsilon_sensitivity.pdf", bbox_inches="tight")
    fig.savefig("epsilon_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_runtime_scaling()
    fig_accuracy_vs_n()
    fig_epsilon_sensitivity()
    print("wrote runtime_scaling.pdf, accuracy_vs_n.pdf, epsilon_sensitivity.pdf (+ .png previews)")
