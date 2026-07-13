"""Downstream-usefulness experiment: next-step forecasting from the latent code.

The reconstruction experiments (src.pca / src.autoencoder / src.vae) measure how
much of X each method can *rebuild*. This module measures something different:
how *useful* the captured information is for a downstream task -- predicting the
next day's return vector.

Protocol (identical across PCA / AE / VAE, so the comparison is fair):
  1. Fit the dimensionality reducer on the TRAIN block only and encode both
     blocks:  Z_train = encode(X_train),  Z_test = encode(X_test).
     (PCA -> transform; AE -> encoder; VAE -> posterior mean mu;
      VAE-t -> posterior location loc.)
  2. Build one-step supervised pairs WITHIN each block:
        features = Z_t          (the current latent)
        target   = X_{t+1}      (the NEXT input vector)      <- target='input'
     Predicting the next *input* (not the next latent) keeps the target
     identical for every method, so the input-space error is directly
     comparable and is not confounded by each method's decoder.
  3. Train a small predictor (linear probe or MLP) on the train pairs,
     early-stopping on a time-ordered validation slice, and evaluate on the
     test pairs.

Reported per k (indexed DataFrame):
    train_pred_mse, test_pred_mse   per-element MSE of the next-step prediction
    train_pred_r2,  test_pred_r2    R^2 against the block mean

Baselines for context (constant in k, drawn as reference lines):
    full-X   an uncompressed predictor X_t -> X_{t+1} (the k = d reference)
    naive    predict the train target mean (the "no skill" R^2 = 0 line)

NOTE: daily log returns are close to unpredictable in the conditional mean
(efficient market), so absolute R^2 will be small for every method. The point
is the RELATIVE ranking of the methods and how it moves with the latent size k.
"""

from __future__ import annotations

from typing import Sequence, Callable

import numpy as np
import pandas as pd

import torch
from torch import nn

from sklearn.decomposition import PCA

from src.pca import COMPONENT_GRID
from src.data import VAL_FRAC
from src import autoencoder as ae_mod
from src import vae as vae_mod
from src import vae_tstudent as vaet_mod


# ----------------------------------------------------------------------------
# Constants -- change these to alter the predictor / training.
# ----------------------------------------------------------------------------
HORIZON = 1                 # predict this many steps ahead

PRED_HIDDEN     = [64]      # predictor hidden sizes; [] == a plain linear probe
PRED_ACTIVATION = "relu"    # key into src.autoencoder.ACTIVATIONS (MLP only)

PRED_LR         = 1e-3
PRED_BATCH      = 64
PRED_MAX_EPOCHS = 2000
PRED_PATIENCE   = 50        # early-stopping patience on validation MSE
PRED_SEED       = 0

DEVICE = ae_mod.DEVICE


# ----------------------------------------------------------------------------
# Predictor model
# ----------------------------------------------------------------------------
class Predictor(nn.Module):
    """MLP mapping in_dim -> out_dim with a linear output.

    hidden=[] collapses to a single linear layer (a ridge-style linear probe);
    hidden=[h1, h2, ...] inserts those hidden layers with `activation` between
    them.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = PRED_HIDDEN,
        activation: str = PRED_ACTIVATION,
    ):
        super().__init__()
        act = ae_mod.ACTIVATIONS[activation]
        sizes = [in_dim, *hidden, out_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:      # linear output layer
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def predictor_label(hidden: Sequence[int]) -> str:
    """Short tag used in logs / result labels."""
    return "linear" if len(hidden) == 0 else "mlp" + str(list(hidden))


# ----------------------------------------------------------------------------
# Supervised pairs + training
# ----------------------------------------------------------------------------
def make_pairs(features: np.ndarray, targets: np.ndarray, horizon: int = HORIZON):
    """(features_t, targets_{t+horizon}) built WITHIN a single contiguous block."""
    return features[:-horizon], targets[horizon:]


def train_predictor(
    Xf: np.ndarray, Yf: np.ndarray,
    Xv: np.ndarray, Yv: np.ndarray,
    hidden: Sequence[int] = PRED_HIDDEN,
    activation: str = PRED_ACTIVATION,
    seed: int = PRED_SEED,
) -> Predictor:
    """Train one predictor with Adam + MSE, early-stopping on validation MSE."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model   = Predictor(Xf.shape[1], Yf.shape[1], hidden, activation).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=PRED_LR)
    loss_fn = nn.MSELoss()

    Xtr = torch.as_tensor(Xf, dtype=torch.float32, device=DEVICE)
    Ytr = torch.as_tensor(Yf, dtype=torch.float32, device=DEVICE)
    Xvl = torch.as_tensor(Xv, dtype=torch.float32, device=DEVICE)
    Yvl = torch.as_tensor(Yv, dtype=torch.float32, device=DEVICE)
    n   = Xtr.shape[0]

    best_val, best_state, wait = float("inf"), None, 0
    for _ in range(PRED_MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, PRED_BATCH):
            idx = perm[i:i + PRED_BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xvl), Yvl).item()
        if vloss < best_val - 1e-8:
            best_val, best_state, wait = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PRED_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict(model: Predictor, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _mse_r2(Y: np.ndarray, Yhat: np.ndarray) -> tuple[float, float]:
    """Per-element MSE and R^2 (against Y's own column means)."""
    ss_res = ((Y - Yhat) ** 2).sum()
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum()
    mse = float(np.mean((Y - Yhat) ** 2))
    r2  = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mse, r2


# ----------------------------------------------------------------------------
# Latent providers: fit a reducer on train, return (Z_train, Z_test)
# ----------------------------------------------------------------------------
def pca_latents(data: dict, k: int, **_) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=k, svd_solver="full").fit(data["X_train"])
    return pca.transform(data["X_train"]), pca.transform(data["X_test"])


def ae_latents(data: dict, k: int, **kw) -> tuple[np.ndarray, np.ndarray]:
    model = ae_mod.train_autoencoder(
        data["X_fit"], data["X_val"], data["d"], k,
        encoder_hidden=kw.get("encoder_hidden", ae_mod.DEFAULT_ENCODER_HIDDEN),
        decoder_hidden=kw.get("decoder_hidden", ae_mod.DEFAULT_DECODER_HIDDEN),
        activation=kw.get("activation", ae_mod.DEFAULT_ACTIVATION),
        seed=kw.get("seed", ae_mod.SEED),
    )
    model.eval()
    with torch.no_grad():
        z_tr = model.encoder(torch.as_tensor(data["X_train"], dtype=torch.float32, device=DEVICE)).cpu().numpy()
        z_te = model.encoder(torch.as_tensor(data["X_test"],  dtype=torch.float32, device=DEVICE)).cpu().numpy()
    return z_tr, z_te


def vae_latents(data: dict, k: int, **kw) -> tuple[np.ndarray, np.ndarray]:
    model = vae_mod.train_vae(
        data["X_fit"], data["X_val"], data["d"], k,
        encoder_hidden=kw.get("encoder_hidden", vae_mod.DEFAULT_ENCODER_HIDDEN),
        decoder_hidden=kw.get("decoder_hidden", vae_mod.DEFAULT_DECODER_HIDDEN),
        activation=kw.get("activation", vae_mod.DEFAULT_ACTIVATION),
        beta=kw.get("beta", vae_mod.DEFAULT_BETA),
        seed=kw.get("seed", vae_mod.SEED),
    )
    model.eval()
    with torch.no_grad():
        mu_tr, _ = model.encode(torch.as_tensor(data["X_train"], dtype=torch.float32, device=DEVICE))
        mu_te, _ = model.encode(torch.as_tensor(data["X_test"],  dtype=torch.float32, device=DEVICE))
    return mu_tr.cpu().numpy(), mu_te.cpu().numpy()


def vaet_latents(data: dict, k: int, **kw) -> tuple[np.ndarray, np.ndarray]:
    model = vaet_mod.train_vae_tstudent(
        data["X_fit"], data["X_val"], data["d"], k,
        encoder_hidden=kw.get("encoder_hidden", vaet_mod.DEFAULT_ENCODER_HIDDEN),
        decoder_hidden=kw.get("decoder_hidden", vaet_mod.DEFAULT_DECODER_HIDDEN),
        activation=kw.get("activation", vaet_mod.DEFAULT_ACTIVATION),
        beta=kw.get("beta", vaet_mod.DEFAULT_BETA),
        nu=kw.get("nu", vaet_mod.DEFAULT_NU),
        learn_nu=kw.get("learn_nu", vaet_mod.DEFAULT_LEARN_NU),
        seed=kw.get("seed", vaet_mod.SEED),
    )
    model.eval()
    with torch.no_grad():
        loc_tr, _ = model.encode(torch.as_tensor(data["X_train"], dtype=torch.float32, device=DEVICE))
        loc_te, _ = model.encode(torch.as_tensor(data["X_test"],  dtype=torch.float32, device=DEVICE))
    return loc_tr.cpu().numpy(), loc_te.cpu().numpy()


LATENT_PROVIDERS: dict[str, Callable] = {
    "PCA":   pca_latents,
    "AE":    ae_latents,
    "VAE":   vae_latents,
    "VAE-t": vaet_latents,
}


# ----------------------------------------------------------------------------
# Core: fit a predictor on given latents and score it
# ----------------------------------------------------------------------------
def _score_predictor(
    Z_train: np.ndarray, Z_test: np.ndarray,
    data: dict, hidden: Sequence[int], activation: str, seed: int,
) -> dict:
    """Build next-input pairs from the latents, train, and return metrics."""
    X_train, X_test = data["X_train"], data["X_test"]

    # Standardise the latent FEATURES using train-block statistics. PCA scores
    # carry the eigenvalue as their variance (PC1 -- the market factor -- is
    # O(10)), whereas AE/VAE latents are O(1); without this the predictor would
    # train on wildly different input scales across methods and the comparison
    # would be unfair. Rescaling is linear, so it preserves the information the
    # probe is meant to measure.
    mu = Z_train.mean(axis=0)
    sd = Z_train.std(axis=0)
    sd[sd == 0] = 1.0
    Z_train = (Z_train - mu) / sd
    Z_test  = (Z_test  - mu) / sd

    # One-step pairs, built within each block (never straddling the split).
    Ztr, Ytr = make_pairs(Z_train, X_train, HORIZON)
    Zte, Yte = make_pairs(Z_test,  X_test,  HORIZON)

    # Time-ordered validation slice from the tail of the train pairs.
    n_val = max(1, int(round(VAL_FRAC * Ztr.shape[0])))
    Zf, Yf = Ztr[:-n_val], Ytr[:-n_val]
    Zv, Yv = Ztr[-n_val:], Ytr[-n_val:]

    model = train_predictor(Zf, Yf, Zv, Yv, hidden=hidden, activation=activation, seed=seed)

    train_mse, train_r2 = _mse_r2(Ytr, _predict(model, Ztr))
    test_mse,  test_r2  = _mse_r2(Yte, _predict(model, Zte))
    return {
        "train_pred_mse": train_mse, "test_pred_mse": test_mse,
        "train_pred_r2":  train_r2,  "test_pred_r2":  test_r2,
    }


# ----------------------------------------------------------------------------
# Experiment runner (one method, swept over k)
# ----------------------------------------------------------------------------
def run_prediction_experiment(
    data: dict,
    method: str = "PCA",
    component_grid: list[int] = COMPONENT_GRID,
    hidden: Sequence[int] = PRED_HIDDEN,
    activation: str = PRED_ACTIVATION,
    encoder_kwargs: dict | None = None,
    seed: int = PRED_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep one method across component_grid, returning a results DataFrame
    indexed by k. `method` is a key into LATENT_PROVIDERS ('PCA'|'AE'|'VAE').

    `encoder_kwargs` is forwarded to the AE / VAE trainer (encoder_hidden,
    decoder_hidden, activation, beta, ...); ignored for PCA.
    """
    provider = LATENT_PROVIDERS[method]
    encoder_kwargs = encoder_kwargs or {}
    tag = predictor_label(hidden)

    rows = []
    for k in component_grid:
        Z_train, Z_test = provider(data, k, **encoder_kwargs)
        m = _score_predictor(Z_train, Z_test, data, hidden, activation, seed)
        if verbose:
            print(f"[predict/{method}/{tag}] k={k:>3}  "
                  f"test R^2={m['test_pred_r2']:+.4f}  test MSE={m['test_pred_mse']:.4f}")
        rows.append({"k": k, **m})

    return pd.DataFrame(rows).set_index("k")


def run_prediction_experiment_multi(
    data: dict,
    method: str = "PCA",
    component_grid: list[int] = COMPONENT_GRID,
    hidden_list: Sequence[Sequence[int]] = ([], PRED_HIDDEN),
    activation: str = PRED_ACTIVATION,
    encoder_kwargs: dict | None = None,
    seed: int = PRED_SEED,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Like run_prediction_experiment, but scores SEVERAL predictors while
    encoding only ONCE per k (the encoder is the expensive part). Returns
    ``{predictor_label: results_df}`` -- e.g. {'linear': df, 'mlp[64]': df}.
    """
    provider = LATENT_PROVIDERS[method]
    encoder_kwargs = encoder_kwargs or {}
    rows: dict[str, list] = {predictor_label(list(h)): [] for h in hidden_list}

    for k in component_grid:
        Z_train, Z_test = provider(data, k, **encoder_kwargs)   # encode once
        for h in hidden_list:
            tag = predictor_label(list(h))
            m = _score_predictor(Z_train, Z_test, data, list(h), activation, seed)
            if verbose:
                print(f"[predict/{method}/{tag}] k={k:>3}  "
                      f"test R^2={m['test_pred_r2']:+.4f}  test MSE={m['test_pred_mse']:.4f}")
            rows[tag].append({"k": k, **m})

    return {tag: pd.DataFrame(r).set_index("k") for tag, r in rows.items()}


# ----------------------------------------------------------------------------
# Baselines (constant in k)
# ----------------------------------------------------------------------------
def full_x_baseline(
    data: dict,
    hidden: Sequence[int] = PRED_HIDDEN,
    activation: str = PRED_ACTIVATION,
    seed: int = PRED_SEED,
) -> dict:
    """Uncompressed reference: predict X_{t+1} from the full X_t (no encoder)."""
    return _score_predictor(data["X_train"], data["X_test"], data, hidden, activation, seed)


def naive_baseline(data: dict) -> dict:
    """No-skill reference: predict the train target mean for every test row."""
    X_train, X_test = data["X_train"], data["X_test"]
    _, Ytr = make_pairs(X_train, X_train, HORIZON)
    _, Yte = make_pairs(X_test,  X_test,  HORIZON)
    mean_pred_tr = np.repeat(Ytr.mean(axis=0, keepdims=True), Ytr.shape[0], axis=0)
    mean_pred_te = np.repeat(Ytr.mean(axis=0, keepdims=True), Yte.shape[0], axis=0)
    train_mse, train_r2 = _mse_r2(Ytr, mean_pred_tr)
    test_mse,  test_r2  = _mse_r2(Yte, mean_pred_te)
    return {
        "train_pred_mse": train_mse, "test_pred_mse": test_mse,
        "train_pred_r2":  train_r2,  "test_pred_r2":  test_r2,
    }
