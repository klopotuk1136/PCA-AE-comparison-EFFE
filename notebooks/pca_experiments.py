"""PCA dimensionality-reduction experiments on S&P 500 log returns.

Varies n_components over {1, 2, 4, 8, 16, 32} and records:
  - preserved variance      (train explained ratio + test R^2)
  - residual entropy        (2nd-order / Gaussian entropy of what PCA discards)
  - reconstruction MSE      (train + test)

A note on the residual entropy: S&P 500 log returns are strongly non-Gaussian
(fat tails, |excess kurtosis| ~ 18 on average in the EDA), so this is a
deliberately SECOND-ORDER measure. It uses 0.5 * (m*log(2*pi*e) + sum log
lambda_i), the entropy of the maximum-entropy (Gaussian) distribution matching
the covariance of the discarded subspace. Read it as "how much linear structure
is still left in what PCA threw away" -- it is an upper bound on the true
fat-tailed entropy, not true information content. It decreases monotonically as
k grows (fewer, smaller residual eigenvalues remain).

Outputs (next to the existing EDA artefacts in ./):
  - pca_experiments.csv
  - plot_pca_experiments.png
  - plain-text table to stdout
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_PATH = r"C:\Users\lenovo\Desktop\UniVie\EFFE\data"
DATA_DIR  = os.path.join(DATA_PATH, "individual_stocks_5yr", "individual_stocks_5yr")
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = 'plots/'

COMPONENT_GRID = [1, 2, 4, 8, 16, 32]
TRAIN_FRAC     = 0.8  # time-based split: first 80% of rows = train


# ----------------------------------------------------------------------------
# Data loading (mirrors the EDA notebook)
# ----------------------------------------------------------------------------
def load_log_returns() -> pd.DataFrame:
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    frames = []
    for f in files:
        df = pd.read_csv(os.path.join(DATA_DIR, f), parse_dates=["date"], index_col="date")
        ticker = df["Name"].iloc[0]
        frames.append(df[["close"]].rename(columns={"close": ticker}))
    prices = pd.concat(frames, axis=1).sort_index()
    log_returns = np.log(prices / prices.shift(1)).iloc[1:]
    return log_returns.dropna(axis=1, how="any")


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def gaussian_entropy_from_eigs(eigs: np.ndarray) -> float:
    """Second-order (Gaussian) differential entropy, in nats, of a distribution
    whose covariance has the given eigenvalues:

        H = 0.5 * (m * log(2*pi*e) + sum_i log(lambda_i))

    For the residual subspace this is the bottom d-k eigenvalues of the
    training covariance."""
    eigs = np.clip(eigs, 1e-300, None)
    m = eigs.size
    if m == 0:
        return 0.0
    return 0.5 * (m * np.log(2.0 * np.pi * np.e) + np.log(eigs).sum())


def reconstruct(pca: PCA, X: np.ndarray) -> np.ndarray:
    """Project X into k-dim latent and back to original space."""
    return pca.inverse_transform(pca.transform(X))


# ----------------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------------
def main() -> None:
    print("Loading log-return matrix...")
    log_returns = load_log_returns()
    print(f"  shape: {log_returns.shape}  (days x stocks)")
    print(f"  range: {log_returns.index[0].date()} -> {log_returns.index[-1].date()}")

    # Time-based split
    n_train = int(round(TRAIN_FRAC * len(log_returns)))
    X_train_raw = log_returns.iloc[:n_train].values
    X_test_raw  = log_returns.iloc[n_train:].values
    print(f"  train rows: {X_train_raw.shape[0]}, test rows: {X_test_raw.shape[0]}")

    # Standardise using train-only statistics (no leakage)
    mu    = X_train_raw.mean(axis=0)
    sigma = X_train_raw.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_train = (X_train_raw - mu) / sigma
    X_test  = (X_test_raw  - mu) / sigma

    # SS-total for the R^2-style test "preserved variance"
    ss_tot_test = ((X_test - X_test.mean(axis=0)) ** 2).sum()

    # Full spectrum on train -- supplies the residual (bottom d-k) eigenvalues
    full_pca = PCA(n_components=None, svd_solver="full").fit(X_train)
    eigvals  = full_pca.explained_variance_  # sorted descending

    rows = []
    for k in COMPONENT_GRID:
        pca = PCA(n_components=k, svd_solver="full").fit(X_train)

        X_train_hat = reconstruct(pca, X_train)
        X_test_hat  = reconstruct(pca, X_test)

        train_mse = np.mean((X_train - X_train_hat) ** 2)
        test_mse  = np.mean((X_test  - X_test_hat)  ** 2)

        # Preserved variance: train uses PCA's explained ratio; test uses R^2.
        train_pres_var = float(pca.explained_variance_ratio_.sum())
        test_pres_var  = 1.0 - ((X_test - X_test_hat) ** 2).sum() / ss_tot_test

        # Residual entropy: 2nd-order entropy of the discarded bottom d-k subspace.
        residual_entropy = gaussian_entropy_from_eigs(eigvals[k:])

        rows.append({
            "k": k,
            "train_preserved_variance": train_pres_var,
            "test_preserved_variance":  test_pres_var,
            "residual_entropy":         residual_entropy,
            "train_mse":                train_mse,
            "test_mse":                 test_mse,
        })

    results = pd.DataFrame(rows).set_index("k")

    # ------ persist ------
    csv_path = os.path.join(OUT_DIR, "pca_experiments.csv")
    results.to_csv(csv_path, float_format="%.6f")
    print(f"\nResults written to {csv_path}\n")

    # ------ print ------
    print(results.round(6).to_string())

    # ------ plot ------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(results.index, results["train_preserved_variance"], "o-", label="train (explained ratio)")
    ax.plot(results.index, results["test_preserved_variance"],  "s-", label="test (R^2)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(COMPONENT_GRID); ax.set_xticklabels(COMPONENT_GRID)
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("Preserved variance")
    ax.set_title("Preserved variance vs k")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(results.index, results["residual_entropy"], "o-", color="C3")
    ax.set_xscale("log", base=2)
    ax.set_xticks(COMPONENT_GRID); ax.set_xticklabels(COMPONENT_GRID)
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("Residual entropy (nats, Gaussian approximation)")
    ax.set_title("Residual entropy vs k")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(results.index, results["train_mse"], "o-", label="train")
    ax.plot(results.index, results["test_mse"],  "s-", label="test")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(COMPONENT_GRID); ax.set_xticklabels(COMPONENT_GRID)
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("Reconstruction MSE (log scale)")
    ax.set_title("Reconstruction MSE vs k")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, PLOTS_DIR+"plot_pca_experiments.png")
    fig.savefig(plot_path, bbox_inches="tight", dpi=120)
    print(f"\nPlot written to {plot_path}")


if __name__ == "__main__":
    main()
