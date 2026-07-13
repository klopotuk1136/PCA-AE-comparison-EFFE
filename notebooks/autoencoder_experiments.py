"""Autoencoder dimensionality-reduction experiments on S&P 500 log returns.

Parallel to pca_experiments.py -- same data, same train-only standardisation,
same time-based split, same component grid {1, 2, 4, 8, 16, 32}, and the same
three metrics, so the autoencoder and PCA results are directly comparable.

Architecture per bottleneck size k:
    [input_dim, 64, k, 64, input_dim]
A CONFIGURABLE activation (sigmoid / relu / tanh) is applied to every hidden
layer (the two 64-unit layers AND the k bottleneck); the OUTPUT layer is always
linear so the net can reproduce signed, standardised targets (a sigmoid output,
bounded to (0,1), could not).

Metrics recorded per k:
  - preserved variance   R^2 = 1 - SS_res/SS_tot, on train and test. (For the
                         AE there is no explained_variance_ratio_, so train uses
                         R^2 too -- unlike the PCA script whose train column is
                         the PCA explained ratio. Both are "fraction of variance
                         reconstructed", so they remain comparable.)
  - residual entropy     2nd-order (Gaussian) differential entropy of the
                         residual X - X_hat. Identical estimator to the PCA
                         script, but here computed from the EMPIRICAL covariance
                         of the (full-dimensional) AE residual on the train block
                         (n_train > d, so it is full rank). It is an upper bound
                         on the true fat-tailed entropy -- "how much linear
                         structure is still left in what the AE failed to keep".
  - reconstruction MSE   mean squared residual, on train and test.

Usage:
  python autoencoder_experiments.py                      # default activation (sigmoid)
  python autoencoder_experiments.py --activation relu    # single activation
  python autoencoder_experiments.py --activation tanh

To sweep all three activations and draw a comparison plot, use the companion
wrapper run_activations.py.

Outputs per activation <act> (next to the existing EDA artefacts in ./):
  - autoencoder_experiments_<act>.csv
  - plot_autoencoder_experiments_<act>.png
  - plain-text table to stdout
"""

import os
import copy
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
# DATA_PATH = r"C:\Users\lenovo\Desktop\UniVie\EFFE\data"
DATA_PATH = r"C:\Users\masha\Data Science\2 semester\EFFE\PCA-AE-comparison-EFFE\data"
DATA_DIR  = os.path.join(DATA_PATH, "individual_stocks_5yr", "individual_stocks_5yr")
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

COMPONENT_GRID = [1, 2, 4, 8, 16, 32]
TRAIN_FRAC     = 0.8   # time-based split: first 80% of rows = train
VAL_FRAC       = 0.10  # fraction of the TRAIN block held out (time-ordered) for early stopping

# Training hyperparameters
HIDDEN     = 64
LR         = 5e-4
BATCH      = 64
MAX_EPOCHS = 5000
PATIENCE   = 100        # early-stopping patience on validation MSE
SEED       = 0

# Configurable hidden activation. nn.* class is instantiated fresh per layer.
ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "relu":    nn.ReLU,
    "tanh":    nn.Tanh,
}
DEFAULT_ACTIVATION = "sigmoid"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Data loading (mirrors the EDA notebook / pca_experiments.py)
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
    """
    eigs = np.clip(eigs, 1e-300, None)
    m = eigs.size
    if m == 0:
        return 0.0
    return 0.5 * (m * np.log(2.0 * np.pi * np.e) + np.log(eigs).sum())


def residual_entropy(residual: np.ndarray) -> float:
    """2nd-order entropy of a full-dimensional residual matrix (rows = samples)
    via the eigenvalues of its empirical covariance."""
    cov = np.cov(residual, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)  # ascending; tiny negatives clipped in helper
    return gaussian_entropy_from_eigs(eigvals)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class AutoEncoder(nn.Module):
    """[d, 64, k, 64, d] with a configurable hidden activation and linear output."""

    def __init__(self, d: int, k: int, activation: str = DEFAULT_ACTIVATION):
        super().__init__()
        act = ACTIVATIONS[activation]
        self.encoder = nn.Sequential(
            nn.Linear(d, HIDDEN), act(),
            nn.Linear(HIDDEN, k), act(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(k, HIDDEN), act(),
            nn.Linear(HIDDEN, d),   # linear output
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_fit: np.ndarray, X_val: np.ndarray, d: int, k: int,
                      activation: str) -> AutoEncoder:
    """Train one AE with Adam + MSE, early-stopping on validation MSE."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model   = AutoEncoder(d, k, activation).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    Xtr = torch.as_tensor(X_fit, dtype=torch.float32, device=DEVICE)
    Xvl = torch.as_tensor(X_val, dtype=torch.float32, device=DEVICE)
    n   = Xtr.shape[0]

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb  = Xtr[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xvl), Xvl).item()
        if vloss < best_val - 1e-7:
            best_val, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def reconstruct(model: AutoEncoder, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()


# ----------------------------------------------------------------------------
# Data preparation (shared across activations)
# ----------------------------------------------------------------------------
def prepare_data() -> dict:
    """Load, time-split, standardise (train-only stats) and carve a validation
    slice. Returns everything the experiment loop needs, so the data is loaded
    once even when sweeping several activations."""
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
    X_train = ((X_train_raw - mu) / sigma).astype(np.float32)
    X_test  = ((X_test_raw  - mu) / sigma).astype(np.float32)

    # Time-ordered validation slice from the tail of the train block (early stopping only)
    n_val = int(round(VAL_FRAC * X_train.shape[0]))

    return {
        "X_train":      X_train,
        "X_test":       X_test,
        "X_fit":        X_train[:-n_val],
        "X_val":        X_train[-n_val:],
        "ss_tot_train": ((X_train - X_train.mean(axis=0)) ** 2).sum(),
        "ss_tot_test":  ((X_test  - X_test.mean(axis=0))  ** 2).sum(),
        "d":            X_train.shape[1],
    }


# ----------------------------------------------------------------------------
# One experiment = one activation swept over the component grid
# ----------------------------------------------------------------------------
def run_experiment(activation: str, data: dict = None, save: bool = True) -> pd.DataFrame:
    """Train the AE for every k in COMPONENT_GRID with the given hidden
    activation, returning a results DataFrame indexed by k. When save=True also
    writes the per-activation CSV and standalone plot."""
    if data is None:
        data = prepare_data()
    X_train, X_test = data["X_train"], data["X_test"]
    X_fit, X_val    = data["X_fit"], data["X_val"]
    ss_tot_train    = data["ss_tot_train"]
    ss_tot_test     = data["ss_tot_test"]
    d               = data["d"]

    print(f"\n{'#' * 60}\n#  activation = {activation}\n{'#' * 60}")
    rows = []
    for k in COMPONENT_GRID:
        print(f"[{activation}] [k={k}] training [{d}, {HIDDEN}, {k}, {HIDDEN}, {d}] ...")
        model = train_autoencoder(X_fit, X_val, d, k, activation)

        X_train_hat = reconstruct(model, X_train)
        X_test_hat  = reconstruct(model, X_test)

        train_mse = float(np.mean((X_train - X_train_hat) ** 2))
        test_mse  = float(np.mean((X_test  - X_test_hat)  ** 2))

        # Preserved variance as R^2 on each block.
        train_pres_var = 1.0 - ((X_train - X_train_hat) ** 2).sum() / ss_tot_train
        test_pres_var  = 1.0 - ((X_test  - X_test_hat)  ** 2).sum() / ss_tot_test

        # Residual entropy from the train-block residual (n_train > d -> full rank).
        res_entropy = residual_entropy(X_train - X_train_hat)

        print(f"        train R^2={train_pres_var:.4f}  test R^2={test_pres_var:.4f}  "
              f"train MSE={train_mse:.4f}  test MSE={test_mse:.4f}")

        rows.append({
            "k": k,
            "train_preserved_variance": float(train_pres_var),
            "test_preserved_variance":  float(test_pres_var),
            "residual_entropy":         res_entropy,
            "train_mse":                train_mse,
            "test_mse":                 test_mse,
        })

    results = pd.DataFrame(rows).set_index("k")
    print(f"\n[{activation}] results:")
    print(results.round(6).to_string())

    if save:
        csv_path = os.path.join(OUT_DIR, f"autoencoder_experiments_{activation}.csv")
        results.to_csv(csv_path, float_format="%.6f")
        print(f"[{activation}] results written to {csv_path}")
        _plot_single_activation(results, activation)

    return results


def _plot_single_activation(results: pd.DataFrame, activation: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(results.index, results["train_preserved_variance"], "o-", label="train (R^2)")
    ax.plot(results.index, results["test_preserved_variance"],  "s-", label="test (R^2)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(COMPONENT_GRID); ax.set_xticklabels(COMPONENT_GRID)
    ax.set_xlabel("Bottleneck size (k)")
    ax.set_ylabel("Preserved variance")
    ax.set_title("Preserved variance vs k")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(results.index, results["residual_entropy"], "o-", color="C3")
    ax.set_xscale("log", base=2)
    ax.set_xticks(COMPONENT_GRID); ax.set_xticklabels(COMPONENT_GRID)
    ax.set_xlabel("Bottleneck size (k)")
    ax.set_ylabel("Residual entropy (nats, Gaussian approximation)")
    ax.set_title("Residual entropy vs k")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(results.index, results["train_mse"], "o-", label="train")
    ax.plot(results.index, results["test_mse"],  "s-", label="test")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(COMPONENT_GRID); ax.set_xticklabels(COMPONENT_GRID)
    ax.set_xlabel("Bottleneck size (k)")
    ax.set_ylabel("Reconstruction MSE (log scale)")
    ax.set_title("Reconstruction MSE vs k")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(f"Autoencoder ({activation}) — dimensionality-reduction metrics", fontsize=13)
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, f"plot_autoencoder_experiments_{activation}.png")
    fig.savefig(plot_path, bbox_inches="tight", dpi=120)
    print(f"[{activation}] plot written to {plot_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activation", choices=list(ACTIVATIONS),
                        default=DEFAULT_ACTIVATION,
                        help=f"hidden activation (default: {DEFAULT_ACTIVATION})")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    run_experiment(args.activation)


if __name__ == "__main__":
    main()
