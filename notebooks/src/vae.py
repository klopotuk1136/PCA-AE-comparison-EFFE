"""Variational-autoencoder dimensionality-reduction experiments on S&P 500
log returns.

Third method in the comparison, alongside src.pca and src.autoencoder. Same
data, same train-only standardisation, same time-based split, same component
grid, and -- crucially -- the SAME five metrics, so PCA / AE / VAE results drop
straight into src.plotting.plot_experiments_comparison together.

How the VAE differs from the plain autoencoder:
  - the encoder outputs a Gaussian posterior per sample: a mean `mu` and a
    log-variance `logvar`, each of size k (the two are linear heads on top of
    the shared encoder body);
  - during training a latent code is SAMPLED, z = mu + eps * exp(0.5*logvar)
    (the reparameterisation trick), and the loss adds a KL term pulling that
    posterior toward a standard normal prior;
  - the KL weight `beta` is the VAE-specific knob (beta<1 -> closer to a plain
    AE, beta=1 -> standard VAE, beta>1 -> beta-VAE, stronger regularisation).

For every REPORTED metric the latent code is the deterministic posterior mean
`mu` (no sampling), so the reconstruction is directly comparable to PCA / AE.

Training objective (per sample, averaged over the batch):
    L = recon + beta * KL
    recon = sum_j (x_j - x_hat_j)^2          # summed over the d output dims
    KL    = -0.5 * sum_i (1 + logvar_i - mu_i^2 - exp(logvar_i))   # over k dims
The reported train/test MSE is always the per-ELEMENT mean squared error, the
same quantity PCA and the AE report -- independent of the loss reduction above.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

from src.pca import COMPONENT_GRID
# residual_entropy / the activation registry are shared metrics/utilities;
# reuse them rather than duplicating so all three methods stay identical.
from src.autoencoder import ACTIVATIONS, residual_entropy


# ----------------------------------------------------------------------------
# Constants -- change these to alter the default architecture / training.
# Anything passed explicitly to run_vae_experiment / VAE / train_vae overrides
# the default; these are just the values used when nothing is passed.
# ----------------------------------------------------------------------------
DEFAULT_ENCODER_HIDDEN = [64]   # hidden sizes between input and the mu/logvar heads
DEFAULT_DECODER_HIDDEN = [64]   # hidden sizes between the latent z and the output
DEFAULT_ACTIVATION     = "sigmoid"
DEFAULT_BETA           = 1.0    # KL weight (the VAE-specific knob)

BETA_GRID = [0.1, 1.0, 4.0]     # default sweep for the beta comparison

LR         = 5e-4
BATCH      = 64
MAX_EPOCHS = 5000
PATIENCE   = 100        # early-stopping patience on validation loss (ELBO)
SEED       = 0

# logvar is clamped to this symmetric range wherever it is exponentiated, to
# keep exp() from overflowing early in training. The bound is intentionally
# wide (var in [e^-10, e^10]) so it never interferes with normal learning.
LOGVAR_CLAMP = 10.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class VAE(nn.Module):
    """Fully connected VAE with configurable encoder / decoder hidden layers,
    a configurable hidden activation, twin linear mu/logvar heads and a linear
    output.

    Parameters
    ----------
    d : int
        Input dimensionality (number of stocks).
    k : int
        Latent dimensionality.
    encoder_hidden : sequence of int
        Hidden layer sizes between the input and the mu/logvar heads.
    decoder_hidden : sequence of int
        Hidden layer sizes between the latent z and the output.
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

        # Encoder body: d -> encoder_hidden[-1] (activation after each layer).
        encoder_sizes = [d, *encoder_hidden]
        enc_layers = []
        for i in range(len(encoder_sizes) - 1):
            enc_layers.append(nn.Linear(encoder_sizes[i], encoder_sizes[i + 1]))
            enc_layers.append(act())
        self.encoder_body = nn.Sequential(*enc_layers)

        # Twin linear heads producing the posterior parameters (never squashed).
        last = encoder_sizes[-1]
        self.fc_mu     = nn.Linear(last, k)
        self.fc_logvar = nn.Linear(last, k)

        # Decoder: k -> decoder_hidden -> d, linear output layer.
        decoder_sizes = [k, *decoder_hidden, d]
        dec_layers = []
        for i in range(len(decoder_sizes) - 1):
            dec_layers.append(nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]))
            if i < len(decoder_sizes) - 2:
                dec_layers.append(act())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def describe_architecture(d: int, k: int, encoder_hidden, decoder_hidden) -> str:
    """Human-readable architecture string for logging, showing the twin heads."""
    enc = " -> ".join(str(s) for s in [d, *encoder_hidden])
    dec = " -> ".join(str(s) for s in [*decoder_hidden, d])
    return f"[{enc}] -> mu/logvar({k}) -> z({k}) -> [{dec}]"


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------
def vae_loss(x, x_hat, mu, logvar, beta: float):
    """ELBO loss (per sample, averaged over the batch).

    Returns (total, recon, kl) so callers can log the two terms separately.
    recon is summed over the d output dims; kl is summed over the k latent dims.
    """
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]
    logvar = logvar.clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon + beta * kl, recon, kl


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_vae(
    X_fit: np.ndarray,
    X_val: np.ndarray,
    d: int,
    k: int,
    encoder_hidden: Sequence[int] = DEFAULT_ENCODER_HIDDEN,
    decoder_hidden: Sequence[int] = DEFAULT_DECODER_HIDDEN,
    activation: str = DEFAULT_ACTIVATION,
    beta: float = DEFAULT_BETA,
    seed: int = SEED,
) -> VAE:
    """Train one VAE with Adam, early-stopping on the validation ELBO.

    The validation loss uses the deterministic posterior mean `mu` for
    reconstruction (plus the analytic KL), giving a stable early-stopping
    signal that matches how the model is evaluated afterwards.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model   = VAE(d, k, encoder_hidden, decoder_hidden, activation).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)

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
            x_hat, mu, logvar = model(xb)
            loss, _, _ = vae_loss(xb, x_hat, mu, logvar, beta)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            mu_v, logvar_v = model.encode(Xvl)
            x_hat_v = model.decoder(mu_v)
            vloss, _, _ = vae_loss(Xvl, x_hat_v, mu_v, logvar_v, beta)
            vloss = vloss.item()
        if vloss < best_val - 1e-7:
            best_val, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def vae_reconstruct(model: VAE, X: np.ndarray, sample: bool = False) -> np.ndarray:
    """Reconstruct X. By default uses the deterministic posterior mean `mu`
    (comparable to PCA / AE); set sample=True to draw z ~ q(z|x) instead."""
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        mu, logvar = model.encode(x)
        z = model.reparameterize(mu, logvar) if sample else mu
        return model.decoder(z).cpu().numpy()


def _mean_kl(model: VAE, X: np.ndarray) -> float:
    """Average per-sample KL(q(z|x) || N(0, I)) over the rows of X."""
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        mu, logvar = model.encode(x)
        logvar = logvar.clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
        return float(kl.item())


# ----------------------------------------------------------------------------
# Experiment runner
# ----------------------------------------------------------------------------
def run_vae_experiment(
    data: dict,
    component_grid: list[int] = COMPONENT_GRID,
    encoder_hidden: Sequence[int] = DEFAULT_ENCODER_HIDDEN,
    decoder_hidden: Sequence[int] = DEFAULT_DECODER_HIDDEN,
    activation: str = DEFAULT_ACTIVATION,
    beta: float = DEFAULT_BETA,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep the VAE across component_grid and return a results DataFrame
    indexed by k.

    The first five columns match src.pca.run_pca_experiment /
    src.autoencoder.run_ae_experiment exactly (so the result plugs straight
    into plot_experiments_comparison). Reconstruction always uses the posterior
    mean. An extra ``train_kl`` column records the average posterior KL on the
    train block -- a VAE-specific diagnostic that the comparison plots ignore.
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
            print(f"[VAE/{activation}/beta={beta}] k={k:>3}  training {arch} ...")
        model = train_vae(
            X_fit, X_val, d, k,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            activation=activation,
            beta=beta,
            seed=seed,
        )

        X_train_hat = vae_reconstruct(model, X_train)   # deterministic (mu)
        X_test_hat  = vae_reconstruct(model, X_test)

        train_mse = float(np.mean((X_train - X_train_hat) ** 2))
        test_mse  = float(np.mean((X_test  - X_test_hat)  ** 2))

        train_pres_var = 1.0 - ((X_train - X_train_hat) ** 2).sum() / ss_tot_train
        test_pres_var  = 1.0 - ((X_test  - X_test_hat)  ** 2).sum() / ss_tot_test

        res_entropy = residual_entropy(X_train - X_train_hat)
        train_kl    = _mean_kl(model, X_train)

        if verbose:
            print(f"[VAE/{activation}/beta={beta}] k={k:>3}  "
                  f"train R^2={train_pres_var:.4f}  test R^2={test_pres_var:.4f}  "
                  f"train MSE={train_mse:.4f}  test MSE={test_mse:.4f}  KL={train_kl:.3f}")

        rows.append({
            "k": k,
            "train_preserved_variance": float(train_pres_var),
            "test_preserved_variance":  float(test_pres_var),
            "residual_entropy":         res_entropy,
            "train_mse":                train_mse,
            "test_mse":                 test_mse,
            "train_kl":                 train_kl,
        })

    return pd.DataFrame(rows).set_index("k")
