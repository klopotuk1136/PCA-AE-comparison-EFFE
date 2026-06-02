"""Autoencoder dimensionality-reduction experiments on S&P 500 log returns.

Mirrors src.pca: same data, same train-only standardisation, same time-based
split, same component grid, and the same three metrics so PCA and AE results
are directly comparable.

Architecture (the encoder/decoder hidden layers are configurable, see
DEFAULT_ENCODER_HIDDEN / DEFAULT_DECODER_HIDDEN):

    encoder:  d  ->  encoder_hidden[0]  ->  ...  ->  encoder_hidden[-1]  ->  k
    decoder:  k  ->  decoder_hidden[0]  ->  ...  ->  decoder_hidden[-1]  ->  d

Every hidden layer (including the k bottleneck) receives the configured
activation. The output layer is always linear, so the net can reproduce
signed, standardised targets (a sigmoid output, bounded to (0, 1), could
not).
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import pandas as pd

import torch
from torch import nn

from src.pca import gaussian_entropy_from_eigs, COMPONENT_GRID


# ----------------------------------------------------------------------------
# Constants -- change these to alter the default architecture / training.
# Anything passed explicitly to run_ae_experiment / AutoEncoder / train_autoencoder
# overrides the default; these are just the values used when nothing is passed.
# ----------------------------------------------------------------------------
DEFAULT_ENCODER_HIDDEN = [64]   # hidden sizes between input and bottleneck
DEFAULT_DECODER_HIDDEN = [64]   # hidden sizes between bottleneck and output
DEFAULT_ACTIVATION     = "sigmoid"

LR         = 5e-4
BATCH      = 64
MAX_EPOCHS = 5000
PATIENCE   = 100        # early-stopping patience on validation MSE
SEED       = 0

ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "relu":    nn.ReLU,
    "tanh":    nn.Tanh,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class AutoEncoder(nn.Module):
    """Fully connected autoencoder with configurable encoder / decoder hidden
    layers and a configurable hidden activation (linear output).

    Parameters
    ----------
    d : int
        Input dimensionality (number of stocks).
    k : int
        Bottleneck (latent) dimensionality.
    encoder_hidden : sequence of int
        Hidden layer sizes between the input and the bottleneck.
    decoder_hidden : sequence of int
        Hidden layer sizes between the bottleneck and the output.
    activation : str
        Key into ACTIVATIONS ('sigmoid' | 'relu' | 'tanh').
    """

    def __init__(
        self,
        d: int,
        k: int,
        encoder_hidden: Sequence[int] = DEFAULT_ENCODER_HIDDEN,
        decoder_hidden: Sequence[int] = DEFAULT_DECODER_HIDDEN,
        activation: str = DEFAULT_ACTIVATION,
    ):
        super().__init__()
        act = ACTIVATIONS[activation]

        encoder_sizes = [d, *encoder_hidden, k]
        enc_layers = []
        for i in range(len(encoder_sizes) - 1):
            enc_layers.append(nn.Linear(encoder_sizes[i], encoder_sizes[i + 1]))
            enc_layers.append(act())              # bottleneck also gets the activation
        self.encoder = nn.Sequential(*enc_layers)

        decoder_sizes = [k, *decoder_hidden, d]
        dec_layers = []
        for i in range(len(decoder_sizes) - 1):
            dec_layers.append(nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]))
            if i < len(decoder_sizes) - 2:        # linear output layer
                dec_layers.append(act())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def describe_architecture(d: int, k: int, encoder_hidden, decoder_hidden) -> str:
    """Human-readable [d, h1, ..., k, ..., h_m, d] string for logging."""
    return "[" + ", ".join(str(s) for s in [d, *encoder_hidden, k, *decoder_hidden, d]) + "]"


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_autoencoder(
    X_fit: np.ndarray,
    X_val: np.ndarray,
    d: int,
    k: int,
    encoder_hidden: Sequence[int] = DEFAULT_ENCODER_HIDDEN,
    decoder_hidden: Sequence[int] = DEFAULT_DECODER_HIDDEN,
    activation: str = DEFAULT_ACTIVATION,
    seed: int = SEED,
) -> AutoEncoder:
    """Train one AE with Adam + MSE, early-stopping on validation MSE."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model   = AutoEncoder(d, k, encoder_hidden, decoder_hidden, activation).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    Xtr = torch.as_tensor(X_fit, dtype=torch.float32, device=DEVICE)
    Xvl = torch.as_tensor(X_val, dtype=torch.float32, device=DEVICE)
    n   = Xtr.shape[0]

    best_val, best_state, wait = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
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


def ae_reconstruct(model: AutoEncoder, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def residual_entropy(residual: np.ndarray) -> float:
    """2nd-order entropy of a full-dimensional residual matrix (rows = samples)
    via the eigenvalues of its empirical covariance. Same estimator as the PCA
    side, but here computed from the empirical residual covariance because the
    AE's residual is not aligned with any eigenbasis of the training cov."""
    cov = np.cov(residual, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)  # ascending; tiny negatives clipped in helper
    return gaussian_entropy_from_eigs(eigvals)


# ----------------------------------------------------------------------------
# Experiment runner
# ----------------------------------------------------------------------------
def run_ae_experiment(
    data: dict,
    component_grid: list[int] = COMPONENT_GRID,
    encoder_hidden: Sequence[int] = DEFAULT_ENCODER_HIDDEN,
    decoder_hidden: Sequence[int] = DEFAULT_DECODER_HIDDEN,
    activation: str = DEFAULT_ACTIVATION,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep the autoencoder across component_grid and return a results
    DataFrame indexed by k.

    Columns mirror src.pca.run_pca_experiment so the two are directly
    comparable. Train preserved variance here is R^2 (the AE has no
    explained-variance ratio), which is still the same "fraction of variance
    reconstructed" quantity.
    """
    X_train, X_test = data["X_train"], data["X_test"]
    X_fit, X_val    = data["X_fit"], data["X_val"]
    ss_tot_train    = data["ss_tot_train"]
    ss_tot_test     = data["ss_tot_test"]
    d               = data["d"]

    rows = []
    for k in component_grid:
        if verbose:
            arch = describe_architecture(d, k, encoder_hidden, decoder_hidden)
            print(f"[AE/{activation}] k={k:>3}  training {arch} ...")
        model = train_autoencoder(
            X_fit, X_val, d, k,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            activation=activation,
            seed=seed,
        )

        X_train_hat = ae_reconstruct(model, X_train)
        X_test_hat  = ae_reconstruct(model, X_test)

        train_mse = float(np.mean((X_train - X_train_hat) ** 2))
        test_mse  = float(np.mean((X_test  - X_test_hat)  ** 2))

        train_pres_var = 1.0 - ((X_train - X_train_hat) ** 2).sum() / ss_tot_train
        test_pres_var  = 1.0 - ((X_test  - X_test_hat)  ** 2).sum() / ss_tot_test

        res_entropy = residual_entropy(X_train - X_train_hat)

        if verbose:
            print(f"[AE/{activation}] k={k:>3}  train R^2={train_pres_var:.4f}  "
                  f"test R^2={test_pres_var:.4f}  "
                  f"train MSE={train_mse:.4f}  test MSE={test_mse:.4f}")

        rows.append({
            "k": k,
            "train_preserved_variance": float(train_pres_var),
            "test_preserved_variance":  float(test_pres_var),
            "residual_entropy":         res_entropy,
            "train_mse":                train_mse,
            "test_mse":                 test_mse,
        })

    return pd.DataFrame(rows).set_index("k")
