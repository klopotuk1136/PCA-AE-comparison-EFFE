"""PCA dimensionality-reduction experiments on S&P 500 log returns.

For a grid of latent sizes k, fits PCA(k) on the standardised train block and
records:
  - preserved variance      (train explained ratio + test R^2)
  - residual entropy        (2nd-order / Gaussian entropy of what PCA discards,
                             computed from the bottom d-k eigenvalues of the
                             train covariance)
  - reconstruction MSE      (train + test)

The "residual entropy" is a deliberately second-order measure -- log returns
are fat-tailed, so this is the entropy of the maximum-entropy distribution
matching the discarded subspace's covariance. It is an upper bound on the
true entropy, and decreases monotonically as k grows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ----------------------------------------------------------------------------
# Constants -- change these to sweep a different latent grid.
# ----------------------------------------------------------------------------
COMPONENT_GRID = [1, 2, 4, 8, 16, 32]


# ----------------------------------------------------------------------------
# Shared helpers (also used by the autoencoder module)
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


def pca_reconstruct(pca: PCA, X: np.ndarray) -> np.ndarray:
    """Project X into k-dim latent and back to original space."""
    return pca.inverse_transform(pca.transform(X))


# ----------------------------------------------------------------------------
# Experiment runner
# ----------------------------------------------------------------------------
def run_pca_experiment(
    data: dict,
    component_grid: list[int] = COMPONENT_GRID,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep PCA across component_grid and return a results DataFrame indexed by k.

    `data` is the dict returned by src.data.prepare_data.

    Columns of the returned DataFrame:
        train_preserved_variance   PCA explained-variance ratio summed across k components
        test_preserved_variance    1 - SS_res/SS_tot on the held-out block
        residual_entropy           2nd-order entropy of the discarded bottom d-k subspace
        train_mse, test_mse        mean squared reconstruction error
    """
    X_train, X_test = data["X_train"], data["X_test"]
    ss_tot_test     = data["ss_tot_test"]

    # Full spectrum on train -- supplies the residual (bottom d-k) eigenvalues.
    full_pca = PCA(n_components=None, svd_solver="full").fit(X_train)
    eigvals  = full_pca.explained_variance_  # sorted descending

    rows = []
    for k in component_grid:
        pca = PCA(n_components=k, svd_solver="full").fit(X_train)

        X_train_hat = pca_reconstruct(pca, X_train)
        X_test_hat  = pca_reconstruct(pca, X_test)

        train_mse = float(np.mean((X_train - X_train_hat) ** 2))
        test_mse  = float(np.mean((X_test  - X_test_hat)  ** 2))

        train_pres_var = float(pca.explained_variance_ratio_.sum())
        test_pres_var  = 1.0 - ((X_test - X_test_hat) ** 2).sum() / ss_tot_test

        residual_entropy = gaussian_entropy_from_eigs(eigvals[k:])

        if verbose:
            print(f"[PCA] k={k:>3}  train R^2={train_pres_var:.4f}  "
                  f"test R^2={test_pres_var:.4f}  "
                  f"train MSE={train_mse:.4f}  test MSE={test_mse:.4f}")

        rows.append({
            "k": k,
            "train_preserved_variance": train_pres_var,
            "test_preserved_variance":  float(test_pres_var),
            "residual_entropy":         residual_entropy,
            "train_mse":                train_mse,
            "test_mse":                 test_mse,
        })

    return pd.DataFrame(rows).set_index("k")
