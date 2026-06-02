"""Sweep the autoencoder over all hidden activations and compare them.

Loads the data once, trains the AE for sigmoid / relu / tanh across the
component grid, and overlays the three activations on a single figure. This is
an autoencoder-only comparison (no PCA) -- use run_experiments.py for PCA vs AE.

Convention in the comparison plot:
  - colour    distinguishes the ACTIVATION
  - linestyle distinguishes the SPLIT (test = solid + circle, train = dashed + x)

Usage:
  python run_activations.py             # run all activations, then plot
  python run_activations.py --skip-run  # re-plot from existing per-activation CSVs

Outputs (next to the existing EDA artefacts in ./):
  - autoencoder_experiments_<act>.csv         (one per activation, via the AE script)
  - plot_autoencoder_experiments_<act>.png    (one per activation, via the AE script)
  - plot_activation_comparison.png            (the joint overlay)
"""

import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

import autoencoder_experiments as ae


HERE = os.path.dirname(os.path.abspath(__file__))
ACTIVATIONS = list(ae.ACTIVATIONS)          # ["sigmoid", "relu", "tanh"]
ACT_COLORS  = {"sigmoid": "C0", "relu": "C1", "tanh": "C2"}
PLOTS_DIR = 'plots/'

# ----------------------------------------------------------------------------
# Producing / loading results
# ----------------------------------------------------------------------------
def compute_all() -> dict:
    """Load the data once, then run every activation in-process."""
    data = ae.prepare_data()
    return {act: ae.run_experiment(act, data=data, save=True) for act in ACTIVATIONS}


def load_all() -> dict:
    results = {}
    for act in ACTIVATIONS:
        csv_path = os.path.join(HERE, f"autoencoder_experiments_{act}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{os.path.basename(csv_path)} not found -- run without --skip-run."
            )
        results[act] = pd.read_csv(csv_path, index_col="k")
    return results


# ----------------------------------------------------------------------------
# Joint plotting (colour = activation, linestyle = split)
# ----------------------------------------------------------------------------
def _style_x(ax, grid) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(grid)
    ax.set_xticklabels(grid)
    ax.set_xlabel("Bottleneck size (k)")
    ax.grid(True, alpha=0.3, which="both")


def plot_train_test(ax, data, base, ylabel, title, logy=False) -> None:
    for act, df in data.items():
        c = ACT_COLORS.get(act)
        ax.plot(df.index, df[f"test_{base}"],  "-o", color=c, label=f"{act} test")
        ax.plot(df.index, df[f"train_{base}"], "--x", color=c, alpha=0.6, label=f"{act} train")
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)


def plot_single(ax, data, column, ylabel, title) -> None:
    for act, df in data.items():
        ax.plot(df.index, df[column], "-o", color=ACT_COLORS.get(act), label=act)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)


def make_comparison_figure(data: dict) -> str:
    grid = sorted(set().union(*[df.index.tolist() for df in data.values()]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    plot_train_test(axes[0], data, "preserved_variance",
                    "Preserved variance", "Preserved variance vs k")
    _style_x(axes[0], grid)

    plot_single(axes[1], data, "residual_entropy",
                "Residual entropy (nats, Gaussian approx.)", "Residual entropy vs k")
    _style_x(axes[1], grid)

    plot_train_test(axes[2], data, "mse",
                    "Reconstruction MSE (log scale)", "Reconstruction MSE vs k", logy=True)
    _style_x(axes[2], grid)

    fig.suptitle("Autoencoder activation comparison — sigmoid vs relu vs tanh", fontsize=13)
    fig.tight_layout()

    out_path = os.path.join(HERE, PLOTS_DIR+"plot_activation_comparison.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    return out_path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-run", action="store_true",
                        help="skip training; plot from existing per-activation CSVs")
    args = parser.parse_args()

    data = load_all() if args.skip_run else compute_all()
    out_path = make_comparison_figure(data)

    print(f"\n{'=' * 70}")
    print("Activation comparison (test split):")
    summary = pd.concat(
        {act: df[["test_preserved_variance", "residual_entropy", "test_mse"]]
         for act, df in data.items()},
        axis=1,
    )
    print(summary.round(6).to_string())
    print(f"\nComparison plot written to {out_path}")


if __name__ == "__main__":
    main()
