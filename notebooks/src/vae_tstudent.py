"""Student-t variational-autoencoder dimensionality-reduction experiments on
S&P 500 log returns.

A heavy-tailed sibling of src.vae: identical data, train-only standardisation,
time-based split, component grid and -- crucially -- the SAME five reported
metrics, so PCA / AE / VAE / Student-t VAE results drop straight into
src.plotting.plot_experiments_comparison together. Motivation: daily log returns
are strongly fat-tailed (|excess kurtosis| ~ 18 in the EDA), so a heavy-tailed
latent prior/posterior is better matched to the data than a Gaussian one.

How this differs from the Gaussian src.vae:
  - the latent PRIOR is an independent (diagonal) Student-t,
    p(z) = prod_i StudentT(nu0, 0, 1), instead of a standard normal;
  - the POSTERIOR is an independent Student-t too,
    q(z|x) = prod_i StudentT(nu, loc_i(x), scale_i(x)); the encoder emits a
    location `loc` and a log-scale head (twin linear heads on the shared encoder
    body), exactly where the Gaussian VAE emits mu / logvar;
  - the degrees-of-freedom nu (the tail-heaviness knob) is configurable: pass a
    fixed value (shared by prior and posterior) OR set learn_nu=True to learn a
    single GLOBAL scalar posterior nu (the prior stays fixed at nu0);
  - because the Student-t / Student-t KL has NO closed form, the KL term is a
    single-sample MONTE-CARLO estimate, log q(z|x) - log p(z), evaluated at the
    reparameterised sample z ~ q. torch.distributions.StudentT.rsample is
    pathwise-differentiable (incl. w.r.t. nu), so the reparameterisation trick
    still gives low-variance gradients.

The DECODER and the reconstruction term are UNCHANGED from the Gaussian VAE: a
linear-output decoder and a summed-MSE reconstruction (an implicit unit-variance
Gaussian decoder). Only the LATENT prior/posterior become heavy-tailed, so the
reported reconstruction metrics stay directly comparable to PCA / AE / VAE.

For every REPORTED metric the latent code is the deterministic posterior
location `loc` (no sampling) -- the direct analogue of the Gaussian VAE's mu.

Training objective (per sample, averaged over the batch):
    L      = recon + beta * KL_mc
    recon  = sum_j (x_j - x_hat_j)^2                       # over the d outputs
    KL_mc  = sum_i ( log q(z_i|x) - log p(z_i) ),  z ~ q   # over the k latents
The reported train/test MSE is always the per-ELEMENT mean squared error, the
same quantity PCA / AE / VAE report -- independent of the loss reduction above.
"""

from __future__ import annotations

import copy
import math
from typing import Sequence

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import StudentT

from src.pca import COMPONENT_GRID
# residual_entropy / the activation registry are shared metrics/utilities;
# reuse them rather than duplicating so all methods stay identical.
from src.autoencoder import ACTIVATIONS, residual_entropy


# ----------------------------------------------------------------------------
# Constants -- change these to alter the default architecture / training.
# Anything passed explicitly to run_vae_tstudent_experiment / VAETStudent /
# train_vae_tstudent overrides the default; these are just the fallbacks.
# ----------------------------------------------------------------------------
DEFAULT_ENCODER_HIDDEN = [64]   # hidden sizes between input and the loc/scale heads
DEFAULT_DECODER_HIDDEN = [64]   # hidden sizes between the latent z and the output
DEFAULT_ACTIVATION     = "sigmoid"
DEFAULT_BETA           = 1.0    # KL weight (the VAE-specific knob)

# Degrees of freedom (the Student-t tail-heaviness knob).
DEFAULT_NU       = 3.0          # prior nu0, and posterior nu when it is NOT learned
DEFAULT_LEARN_NU = False        # if True, learn a single GLOBAL scalar posterior nu
NU_FLOOR         = 2.0          # learned nu is floored just above this -> finite variance

BETA_GRID = [0.1, 1.0, 4.0]         # default sweep for the beta comparison
NU_GRID   = [2.5, 3.0, 5.0, 10.0]   # default sweep for the tail (nu) comparison

LR         = 5e-4
BATCH      = 64
MAX_EPOCHS = 5000
PATIENCE   = 100        # early-stopping patience on validation loss (ELBO)
SEED       = 0

# MC samples used ONLY for the (no-grad) validation ELBO and the train_kl
# diagnostic. Training itself uses the standard single-sample estimator. A
# handful of samples makes the early-stopping signal low-variance and stable.
VAL_KL_SAMPLES = 16

# The log-scale head is clamped to this symmetric range wherever it is
# exponentiated, to keep exp() from overflowing early in training. Intentionally
# wide (scale in [e^-10, e^10]) so it never interferes with normal learning.
LOGSCALE_CLAMP = 10.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _inv_softplus(y: float) -> float:
    """Inverse of softplus, used to initialise raw_nu so the learned posterior
    nu starts at a chosen target value (softplus(raw) = y  =>  raw)."""
    return math.log(math.expm1(y))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class VAETStudent(nn.Module):
    """Fully connected Student-t VAE with configurable encoder / decoder hidden
    layers, a configurable hidden activation, twin linear loc/log-scale heads and
    a linear output. The latent prior and posterior are independent (diagonal)
    Student-t distributions.

    Parameters
    ----------
    d : int
        Input dimensionality (number of stocks).
    k : int
        Latent dimensionality.
    encoder_hidden : sequence of int
        Hidden layer sizes between the input and the loc/scale heads.
    decoder_hidden : sequence of int
        Hidden layer sizes between the latent z and the output.
    activation : str
        Key into ACTIVATIONS ('sigmoid' | 'relu' | 'tanh' | 'leakyrelu').
    nu : float
        Degrees of freedom. Always the fixed prior nu0; also the posterior nu
        when learn_nu is False.
    learn_nu : bool
        If True, the posterior nu is a single learnable global scalar,
        parameterised as softplus(raw_nu) + NU_FLOOR and initialised near `nu`.
        The prior nu0 stays fixed at `nu`.
    """

    def __init__(
        self,
        d: int,
        k: int,
        encoder_hidden: Sequence[int] = DEFAULT_ENCODER_HIDDEN,
        decoder_hidden: Sequence[int] = DEFAULT_DECODER_HIDDEN,
        activation: str = DEFAULT_ACTIVATION,
        nu: float = DEFAULT_NU,
        learn_nu: bool = DEFAULT_LEARN_NU,
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
        self.fc_loc      = nn.Linear(last, k)
        self.fc_logscale = nn.Linear(last, k)

        # Decoder: k -> decoder_hidden -> d, linear output layer.
        decoder_sizes = [k, *decoder_hidden, d]
        dec_layers = []
        for i in range(len(decoder_sizes) - 1):
            dec_layers.append(nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]))
            if i < len(decoder_sizes) - 2:
                dec_layers.append(act())
        self.decoder = nn.Sequential(*dec_layers)

        # Degrees of freedom. The prior nu0 is always a fixed hyperparameter.
        self.nu_prior = float(nu)
        self.learn_nu = bool(learn_nu)
        if self.learn_nu:
            target = max(float(nu) - NU_FLOOR, 0.5)   # so posterior nu starts ~ nu
            self.raw_nu = nn.Parameter(torch.tensor(_inv_softplus(target), dtype=torch.float32))
        else:
            # buffer so it moves with .to(DEVICE) and is saved in the state dict
            self.register_buffer("nu_post_buf", torch.tensor(float(nu), dtype=torch.float32))

    def posterior_nu(self) -> torch.Tensor:
        """Current posterior degrees of freedom (learned or fixed), as a tensor."""
        if self.learn_nu:
            return F.softplus(self.raw_nu) + NU_FLOOR
        return self.nu_post_buf

    def _scale(self, logscale: torch.Tensor) -> torch.Tensor:
        return torch.exp(logscale.clamp(-LOGSCALE_CLAMP, LOGSCALE_CLAMP))

    def encode(self, x):
        h = self.encoder_body(x)
        return self.fc_loc(h), self.fc_logscale(h)

    def posterior(self, loc, logscale) -> StudentT:
        """Diagonal Student-t posterior q(z|x)."""
        return StudentT(self.posterior_nu(), loc, self._scale(logscale))

    def prior(self, device) -> StudentT:
        """Standard diagonal Student-t prior p(z) = StudentT(nu0, 0, 1)."""
        zero = torch.zeros((), device=device)
        return StudentT(torch.tensor(self.nu_prior, device=device), zero, zero + 1.0)

    def forward(self, x):
        loc, logscale = self.encode(x)
        z = self.posterior(loc, logscale).rsample()   # reparameterised sample
        return self.decoder(z), loc, logscale, z


def describe_architecture(d, k, encoder_hidden, decoder_hidden, nu, learn_nu) -> str:
    """Human-readable architecture string for logging, showing the twin heads
    and the degrees-of-freedom setting."""
    enc = " -> ".join(str(s) for s in [d, *encoder_hidden])
    dec = " -> ".join(str(s) for s in [*decoder_hidden, d])
    nu_str = "nu=learn" if learn_nu else f"nu={nu}"
    return f"[{enc}] -> loc/scale({k}), {nu_str} -> z({k}) -> [{dec}]"


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------
def vae_tstudent_loss(x, x_hat, z, loc, logscale, nu_post, nu_prior: float, beta: float):
    """Monte-Carlo ELBO loss (per sample, averaged over the batch).

    Returns (total, recon, kl) so callers can log the two terms separately.
    recon is summed over the d output dims; kl is the single-sample MC estimate
    log q(z|x) - log p(z) summed over the k latent dims. Because the KL is a
    stochastic estimate it can be negative for an individual batch -- that is
    expected and unbiased in expectation.
    """
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]

    scale = torch.exp(logscale.clamp(-LOGSCALE_CLAMP, LOGSCALE_CLAMP))
    q = StudentT(nu_post, loc, scale)
    zero = torch.zeros((), device=x.device)
    p = StudentT(torch.as_tensor(float(nu_prior), device=x.device), zero, zero + 1.0)

    kl = (q.log_prob(z) - p.log_prob(z)).sum(dim=1).mean()
    return recon + beta * kl, recon, kl


def _elbo_eval(model: VAETStudent, X: torch.Tensor, beta: float,
               n_kl_samples: int = VAL_KL_SAMPLES) -> torch.Tensor:
    """Validation ELBO used for early stopping. Reconstruction uses the
    deterministic posterior location (matching how the model is evaluated
    afterwards); the KL is an n_kl_samples MC average for a stable signal."""
    loc, logscale = model.encode(X)
    x_hat = model.decoder(loc)
    recon = F.mse_loss(x_hat, X, reduction="sum") / X.shape[0]

    q = model.posterior(loc, logscale)
    p = model.prior(X.device)
    z = q.rsample((n_kl_samples,))                     # (M, n, k)
    kl = (q.log_prob(z) - p.log_prob(z)).sum(dim=-1).mean()
    return recon + beta * kl


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_vae_tstudent(
    X_fit: np.ndarray,
    X_val: np.ndarray,
    d: int,
    k: int,
    encoder_hidden: Sequence[int] = DEFAULT_ENCODER_HIDDEN,
    decoder_hidden: Sequence[int] = DEFAULT_DECODER_HIDDEN,
    activation: str = DEFAULT_ACTIVATION,
    beta: float = DEFAULT_BETA,
    nu: float = DEFAULT_NU,
    learn_nu: bool = DEFAULT_LEARN_NU,
    seed: int = SEED,
) -> VAETStudent:
    """Train one Student-t VAE with Adam, early-stopping on the validation ELBO.

    The validation loss uses the deterministic posterior location for
    reconstruction plus a multi-sample MC KL, giving a stable early-stopping
    signal that matches how the model is evaluated afterwards.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = VAETStudent(d, k, encoder_hidden, decoder_hidden, activation, nu, learn_nu).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

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
            x_hat, loc, logscale, z = model(xb)
            loss, _, _ = vae_tstudent_loss(
                xb, x_hat, z, loc, logscale, model.posterior_nu(), model.nu_prior, beta
            )
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = _elbo_eval(model, Xvl, beta).item()
        if vloss < best_val - 1e-7:
            best_val, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def vae_tstudent_reconstruct(model: VAETStudent, X: np.ndarray, sample: bool = False) -> np.ndarray:
    """Reconstruct X. By default uses the deterministic posterior location `loc`
    (comparable to PCA / AE / VAE); set sample=True to draw z ~ q(z|x) instead."""
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        loc, logscale = model.encode(x)
        z = model.posterior(loc, logscale).rsample() if sample else loc
        return model.decoder(z).cpu().numpy()


def _mean_kl(model: VAETStudent, X: np.ndarray, n_samples: int = VAL_KL_SAMPLES) -> float:
    """Average per-sample MC KL(q(z|x) || p(z)) over the rows of X (diagnostic)."""
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        loc, logscale = model.encode(x)
        q = model.posterior(loc, logscale)
        p = model.prior(x.device)
        z = q.rsample((n_samples,))
        kl = (q.log_prob(z) - p.log_prob(z)).sum(dim=-1).mean()
        return float(kl.item())


# ----------------------------------------------------------------------------
# Experiment runner
# ----------------------------------------------------------------------------
def run_vae_tstudent_experiment(
    data: dict,
    component_grid: list[int] = COMPONENT_GRID,
    encoder_hidden: Sequence[int] = DEFAULT_ENCODER_HIDDEN,
    decoder_hidden: Sequence[int] = DEFAULT_DECODER_HIDDEN,
    activation: str = DEFAULT_ACTIVATION,
    beta: float = DEFAULT_BETA,
    nu: float = DEFAULT_NU,
    learn_nu: bool = DEFAULT_LEARN_NU,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep the Student-t VAE across component_grid and return a results
    DataFrame indexed by k.

    The first five columns match src.pca.run_pca_experiment /
    src.autoencoder.run_ae_experiment / src.vae.run_vae_experiment exactly (so
    the result plugs straight into plot_experiments_comparison). Reconstruction
    always uses the posterior location. Two extra VAE-specific diagnostic columns
    the comparison plots ignore: ``train_kl`` (average posterior MC KL on train)
    and ``nu_value`` (the posterior degrees of freedom actually used -- the
    learned value when learn_nu=True, else the fixed nu).
    """
    X_train, X_test = data["X_train"], data["X_test"]
    X_fit, X_val    = data["X_fit"], data["X_val"]
    ss_tot_train    = data["ss_tot_train"]
    ss_tot_test     = data["ss_tot_test"]
    d               = data["d"]

    rows = []
    for k in component_grid:
        if verbose:
            arch = describe_architecture(d, k, encoder_hidden, decoder_hidden, nu, learn_nu)
            print(f"[VAE-t/{activation}/beta={beta}/nu={'learn' if learn_nu else nu}] "
                  f"k={k:>3}  training {arch} ...")
        model = train_vae_tstudent(
            X_fit, X_val, d, k,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            activation=activation,
            beta=beta,
            nu=nu,
            learn_nu=learn_nu,
            seed=seed,
        )

        X_train_hat = vae_tstudent_reconstruct(model, X_train)   # deterministic (loc)
        X_test_hat  = vae_tstudent_reconstruct(model, X_test)

        train_mse = float(np.mean((X_train - X_train_hat) ** 2))
        test_mse  = float(np.mean((X_test  - X_test_hat)  ** 2))

        train_pres_var = 1.0 - ((X_train - X_train_hat) ** 2).sum() / ss_tot_train
        test_pres_var  = 1.0 - ((X_test  - X_test_hat)  ** 2).sum() / ss_tot_test

        res_entropy = residual_entropy(X_train - X_train_hat)
        train_kl    = _mean_kl(model, X_train)
        nu_value    = float(model.posterior_nu().item())

        if verbose:
            print(f"[VAE-t/{activation}/beta={beta}] k={k:>3}  "
                  f"train R^2={train_pres_var:.4f}  test R^2={test_pres_var:.4f}  "
                  f"train MSE={train_mse:.4f}  test MSE={test_mse:.4f}  "
                  f"KL={train_kl:.3f}  nu={nu_value:.3f}")

        rows.append({
            "k": k,
            "train_preserved_variance": float(train_pres_var),
            "test_preserved_variance":  float(test_pres_var),
            "residual_entropy":         res_entropy,
            "train_mse":                train_mse,
            "test_mse":                 test_mse,
            "train_kl":                 train_kl,
            "nu_value":                 nu_value,
        })

    return pd.DataFrame(rows).set_index("k")
