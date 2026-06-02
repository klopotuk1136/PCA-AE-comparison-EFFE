"""Run both dimensionality-reduction experiments and draw joint comparison plots.

Calls pca_experiments.py and autoencoder_experiments.py (each writes its own CSV
and standalone plot), then overlays the two methods on a single figure so PCA and
the autoencoder can be compared metric-by-metric across the component grid.

Convention in the joint plots:
  - colour   distinguishes the METHOD   (PCA vs autoencoder)
  - linestyle distinguishes the SPLIT    (test = solid + circle, train = dashed + x)

Usage:
  python run_experiments.py            # run both scripts, then plot
  python run_experiments.py --skip-run # re-plot from existing CSVs only

Outputs (next to the existing EDA artefacts in ./):
  - plot_combined_experiments.png
"""

import os
import sys
import argparse
import subprocess

import pandas as pd
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = 'plots/'

# Which autoencoder activation to compare against PCA here.
AE_ACTIVATION = "relu"

EXPERIMENTS = {
    # method label   (script,                       extra CLI args,                results CSV)
    "PCA":         ("pca_experiments.py",         [],                            "pca_experiments.csv"),
    "Autoencoder": ("autoencoder_experiments.py", ["--activation", AE_ACTIVATION],
                    f"autoencoder_experiments_{AE_ACTIVATION}.csv"),
}

METHOD_STYLE = {
    "PCA":         {"color": "C0"},
    "Autoencoder": {"color": "C1"},
}


# ----------------------------------------------------------------------------
# Running the child experiment scripts
# ----------------------------------------------------------------------------
def run_script(script: str, script_args: list) -> None:
    """Execute a sibling script with the SAME interpreter running this wrapper."""
    path = os.path.join(HERE, script)
    print(f"\n{'=' * 70}\n  running {script} {' '.join(script_args)}\n{'=' * 70}")
    subprocess.run([sys.executable, path, *script_args], check=True, cwd=HERE)


def load_results() -> dict:
    """Read each method's results CSV into a DataFrame indexed by k."""
    data = {}
    for method, (_script, _args, csv_name) in EXPERIMENTS.items():
        csv_path = os.path.join(HERE, csv_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{csv_name} not found -- run without --skip-run to generate it."
            )
        data[method] = pd.read_csv(csv_path, index_col="k")
    return data


# ----------------------------------------------------------------------------
# Joint plotting
# ----------------------------------------------------------------------------
def _style_x(ax, grid) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(grid)
    ax.set_xticklabels(grid)
    ax.set_xlabel("Latent / component size (k)")
    ax.grid(True, alpha=0.3, which="both")


def plot_train_test(ax, data, base, ylabel, title, logy=False) -> None:
    """Overlay a train/test metric pair (columns train_<base> / test_<base>)."""
    for method, df in data.items():
        c = METHOD_STYLE[method]["color"]
        ax.plot(df.index, df[f"test_{base}"],  "-o", color=c, label=f"{method} test")
        ax.plot(df.index, df[f"train_{base}"], "--x", color=c, alpha=0.6, label=f"{method} train")
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)


def plot_single(ax, data, column, ylabel, title) -> None:
    """Overlay a single-column metric (e.g. residual entropy)."""
    for method, df in data.items():
        c = METHOD_STYLE[method]["color"]
        ax.plot(df.index, df[column], "-o", color=c, label=method)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)


def make_joint_figure(data: dict) -> str:
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

    fig.suptitle("PCA vs Autoencoder — dimensionality-reduction metrics", fontsize=13)
    fig.tight_layout()

    out_path = os.path.join(HERE, PLOTS_DIR+"plot_combined_experiments.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    return out_path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-run", action="store_true",
                        help="skip running the experiment scripts; plot from existing CSVs")
    args = parser.parse_args()

    if not args.skip_run:
        for method, (script, script_args, _csv) in EXPERIMENTS.items():
            run_script(script, script_args)

    data = load_results()
    out_path = make_joint_figure(data)

    print(f"\n{'=' * 70}")
    print("Combined results (test split):")
    summary = pd.concat(
        {m: df[["test_preserved_variance", "residual_entropy", "test_mse"]]
         for m, df in data.items()},
        axis=1,
    )
    print(summary.round(6).to_string())
    print(f"\nJoint plot written to {out_path}")


if __name__ == "__main__":
    main()
