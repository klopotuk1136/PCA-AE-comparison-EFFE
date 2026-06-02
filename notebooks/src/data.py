"""Data loading for the S&P 500 log-return experiments.

Single source of truth for the dataset, the train/test split, and the
standardisation. The EDA notebook and every experiment imports from here so
they share the exact same numbers.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Constants -- change these to point at a different dataset or to change the
# train / val split that every experiment uses.
# ----------------------------------------------------------------------------
DATA_PATH = r"C:\Users\lenovo\Desktop\UniVie\EFFE\data"
DATA_DIR  = os.path.join(DATA_PATH, "individual_stocks_5yr", "individual_stocks_5yr")

TRAIN_FRAC = 0.8    # time-based split: first 80% of rows = train
VAL_FRAC   = 0.10   # fraction of the TRAIN block held out (time-ordered) for early stopping
SEED       = 0


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_prices(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Wide close-price matrix (days x stocks). Columns are ticker names."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    frames = []
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f), parse_dates=["date"], index_col="date")
        ticker = df["Name"].iloc[0]
        frames.append(df[["close"]].rename(columns={"close": ticker}))
    return pd.concat(frames, axis=1).sort_index()


def load_log_returns(data_dir: str = DATA_DIR, complete_only: bool = True) -> pd.DataFrame:
    """Daily log-return matrix derived from load_prices().

    complete_only=True (the default used by every experiment) drops any stock
    that has even one missing return, which yields a fully populated matrix
    suitable for PCA / autoencoder training.
    """
    prices = load_prices(data_dir)
    log_returns = np.log(prices / prices.shift(1)).iloc[1:]
    if complete_only:
        log_returns = log_returns.dropna(axis=1, how="any")
    return log_returns


# ----------------------------------------------------------------------------
# Train / test split + standardisation (train-only stats -> no leakage)
# ----------------------------------------------------------------------------
def prepare_data(
    log_returns: pd.DataFrame | None = None,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> dict:
    """Return everything an experiment needs in a single dict.

    The returned dict has:
        X_train, X_test          standardised (n_train, d) / (n_test, d) arrays
        X_fit, X_val             time-ordered split of X_train for early stopping
        ss_tot_train, ss_tot_test sum-of-squares totals for R^2-style metrics
        d                        number of stocks
        mu, sigma                train-only column statistics used to standardise
        log_returns              the underlying DataFrame (handy for plotting)
    """
    if log_returns is None:
        log_returns = load_log_returns()

    n_train = int(round(train_frac * len(log_returns)))
    X_train_raw = log_returns.iloc[:n_train].values
    X_test_raw  = log_returns.iloc[n_train:].values

    mu    = X_train_raw.mean(axis=0)
    sigma = X_train_raw.std(axis=0)
    sigma[sigma == 0] = 1.0

    X_train = ((X_train_raw - mu) / sigma).astype(np.float32)
    X_test  = ((X_test_raw  - mu) / sigma).astype(np.float32)

    n_val = int(round(val_frac * X_train.shape[0]))
    X_fit = X_train[:-n_val] if n_val > 0 else X_train
    X_val = X_train[-n_val:] if n_val > 0 else X_train

    return {
        "X_train":      X_train,
        "X_test":       X_test,
        "X_fit":        X_fit,
        "X_val":        X_val,
        "ss_tot_train": float(((X_train - X_train.mean(axis=0)) ** 2).sum()),
        "ss_tot_test":  float(((X_test  - X_test.mean(axis=0))  ** 2).sum()),
        "d":            X_train.shape[1],
        "mu":           mu,
        "sigma":        sigma,
        "log_returns":  log_returns,
    }
