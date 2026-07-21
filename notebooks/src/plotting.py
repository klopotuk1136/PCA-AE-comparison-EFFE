"""Standardised plots for the EDA notebook and the experiments notebook.

Every plot is a thin wrapper around matplotlib that takes already-computed
inputs and returns a Figure. Save paths are optional -- if `save_path` is
given the figure is written to disk, otherwise it is just returned for
display in the notebook.

The headline function for the experiments notebook is
`plot_experiments_comparison`: pass it a `{label: results_df}` dict and it
overlays every method / variant on a single three-panel figure (preserved
variance, residual entropy, reconstruction MSE), with colour distinguishing
the label and linestyle distinguishing train vs test.
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats


# ----------------------------------------------------------------------------
# Constants -- change these to redirect plot output or tweak global style.
# ----------------------------------------------------------------------------
PLOTS_DIR = "plots/"
FIG_DPI   = 120

# Colours used to distinguish experiments when no explicit style dict is given.
DEFAULT_COLOR_CYCLE = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


def setup_style() -> None:
    """Call once at the top of a notebook to standardise the look."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.dpi"] = FIG_DPI


def _save(fig: plt.Figure, save_path: str | None) -> None:
    if save_path is None:
        return
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=FIG_DPI)


# ============================================================================
# EDA plots
# ============================================================================
def plot_price_vs_returns(
    prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    picks: Mapping[str, str],
    save_path: str | None = None,
) -> plt.Figure:
    """For each (ticker, sector) in `picks`, plot the close price and the
    daily log-return side by side."""
    fig, axes = plt.subplots(len(picks), 2, figsize=(14, 3.5 * len(picks)))
    if len(picks) == 1:
        axes = np.atleast_2d(axes)

    for i, (ticker, sector) in enumerate(picks.items()):
        p = prices[ticker].dropna()
        r = log_returns[ticker].dropna()
        ax_p, ax_r = axes[i]

        ax_p.plot(p.index, p.values, linewidth=1)
        ax_p.set_title(f"{ticker} ({sector}) -- Close Price", fontsize=10)
        ax_p.set_ylabel("USD")
        ax_p.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

        ax_r.plot(r.index, r.values, linewidth=0.6, alpha=0.8)
        ax_r.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax_r.set_title(f"{ticker} ({sector}) -- Daily Log Return", fontsize=10)
        ax_r.set_ylabel("Log return")
        ax_r.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_return_distributions(
    log_returns: pd.DataFrame,
    picks: Iterable[str],
    save_path: str | None = None,
) -> plt.Figure:
    """Per-ticker histogram of log returns with a fitted Normal overlay."""
    picks = list(picks)
    fig, axes = plt.subplots(1, len(picks), figsize=(14, 4), sharey=False)
    if len(picks) == 1:
        axes = [axes]

    for ax, ticker in zip(axes, picks):
        r = log_returns[ticker].dropna()
        mu, sigma = r.mean(), r.std()

        ax.hist(r, bins=60, density=True, alpha=0.6, label="Empirical")
        x = np.linspace(r.min(), r.max(), 300)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=1.5, label="Normal fit")

        kurt = stats.kurtosis(r)
        ax.set_title(f"{ticker}\nmu={mu:.4f}  sigma={sigma:.4f}\nExcess kurtosis={kurt:.2f}",
                     fontsize=9)
        ax.set_xlabel("Log return")
        ax.legend(fontsize=8)

    fig.suptitle("Daily Log-Return Distributions", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_rolling_volatility(
    log_returns: pd.DataFrame,
    picks: Iterable[str],
    window: int = 30,
    trading_days_per_year: int = 252,
    save_path: str | None = None,
) -> plt.Figure:
    """Rolling annualised volatility for the tickers in `picks`."""
    fig, ax = plt.subplots(figsize=(13, 4))
    for ticker in picks:
        rolling_vol = log_returns[ticker].rolling(window).std() * np.sqrt(trading_days_per_year)
        ax.plot(rolling_vol.index, rolling_vol.values, linewidth=1, label=ticker)

    ax.set_title(f"Rolling {window}-day Annualised Volatility", fontsize=11)
    ax.set_ylabel("Volatility (annualised)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_pairwise_correlation_distribution(
    corr_matrix: pd.DataFrame,
    save_path: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Histogram of the upper-triangle entries of `corr_matrix`.
    Returns the figure AND the raw pairwise correlation values."""
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    pairwise_corrs = upper.stack().values

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pairwise_corrs, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(pairwise_corrs.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {pairwise_corrs.mean():.2f}")
    ax.set_title("Distribution of Pairwise Return Correlations (all S&P 500 pairs)", fontsize=11)
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig, pairwise_corrs


def plot_correlation_heatmap(
    log_returns: pd.DataFrame,
    sector_samples: Mapping[str, Iterable[str]],
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap of pairwise correlations for a sector-grouped subset of tickers.
    Sector boundaries are drawn as black lines."""
    selected = [t for tickers in sector_samples.values() for t in tickers if t in log_returns.columns]
    sub_corr = log_returns[selected].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        sub_corr, ax=ax,
        cmap="RdYlGn", center=0, vmin=-0.2, vmax=1,
        xticklabels=selected, yticklabels=selected,
        linewidths=0.3, linecolor="white",
    )
    ax.set_title(
        f"Return Correlation Heatmap -- {len(sector_samples)} Sectors x "
        f"{max(len(list(t)) for t in sector_samples.values())} Stocks",
        fontsize=12,
    )
    ax.tick_params(axis="both", labelsize=7)

    sector_sizes = [sum(1 for t in tickers if t in log_returns.columns)
                    for tickers in sector_samples.values()]
    for b in np.cumsum(sector_sizes)[:-1]:
        ax.axhline(b, color="black", linewidth=1.5)
        ax.axvline(b, color="black", linewidth=1.5)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_pca_scree(
    explained_variance_ratio: np.ndarray,
    n_show: int = 50,
    thresholds: Iterable[float] = (0.80, 0.90, 0.95),
    xlim_cumulative: int = 150,
    save_path: str | None = None,
) -> plt.Figure:
    """Two-panel PCA variance plot: scree (left) + cumulative explained variance
    with the n_components needed to hit each `thresholds` value marked."""
    cumulative = np.cumsum(explained_variance_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(range(1, n_show + 1), explained_variance_ratio[:n_show] * 100, color="steelblue")
    axes[0].set_title("Scree Plot -- Variance per Component", fontsize=11)
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance (%)")

    axes[1].plot(range(1, len(cumulative) + 1), cumulative * 100, linewidth=1.5)
    for threshold in thresholds:
        n_comp = int(np.argmax(cumulative >= threshold) + 1)
        axes[1].axhline(threshold * 100, color="red", linestyle="--", linewidth=0.8)
        axes[1].axvline(n_comp, color="red", linestyle="--", linewidth=0.8)
        axes[1].annotate(f"{int(threshold * 100)}% -> {n_comp} PCs",
                         xy=(n_comp, threshold * 100),
                         xytext=(n_comp + 5, threshold * 100 - 4),
                         fontsize=8, color="red")

    axes[1].set_title("Cumulative Explained Variance", fontsize=11)
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_xlim(0, xlim_cumulative)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_pca_2d(
    coords: np.ndarray,
    years: Iterable[int],
    explained_variance_ratio: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Scatter of all days in (PC1, PC2), coloured by year."""
    years = np.asarray(list(years))
    unique_years = sorted(np.unique(years))
    palette = sns.color_palette("tab10", n_colors=len(unique_years))
    year_to_color = {y: palette[i] for i, y in enumerate(unique_years)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for year in unique_years:
        mask = years == year
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[year_to_color[year]], s=8, alpha=0.6, label=str(year))

    ax.set_title(
        f"Days projected onto PC1 & PC2\n"
        f"(PC1={explained_variance_ratio[0] * 100:.1f}%, "
        f"PC2={explained_variance_ratio[1] * 100:.1f}%)",
        fontsize=11,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Year", markerscale=2, fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_summary_stats(
    summary: pd.DataFrame,
    columns: Iterable[str] = ("std", "skewness", "kurtosis"),
    xlabels: Iterable[str] = ("Daily std of log returns", "Skewness", "Excess kurtosis"),
    save_path: str | None = None,
) -> plt.Figure:
    """Histogram of the per-stock statistics in `columns`."""
    columns = list(columns); xlabels = list(xlabels)
    fig, axes = plt.subplots(1, len(columns), figsize=(13, 4))

    for ax, col, xlabel in zip(axes, columns, xlabels):
        ax.hist(summary[col], bins=40, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(summary[col].mean(), color="red", linestyle="--", linewidth=1.3,
                   label=f"mean={summary[col].mean():.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of stocks")
        ax.legend(fontsize=8)

    fig.suptitle("Distribution of Per-Stock Return Statistics (470 stocks)", fontsize=11)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ============================================================================
# Experiment comparison plot -- the workhorse for the experiments notebook.
# ============================================================================
def plot_experiments_comparison(
    experiments: Mapping[str, pd.DataFrame],
    title: str = "Dimensionality-reduction comparison",
    styles: Mapping[str, Mapping] | None = None,
    show_train: bool = True,
    save_path: str | None = None,
    figsize: tuple = (15, 4.2),
) -> plt.Figure:
    """Overlay an arbitrary number of experiments on one three-panel figure.

    Parameters
    ----------
    experiments
        Dict mapping label -> results DataFrame (the kind produced by
        run_pca_experiment / run_ae_experiment). Each DataFrame is indexed by k
        and must contain the columns ``train_preserved_variance``,
        ``test_preserved_variance``, ``residual_entropy``, ``train_mse`` and
        ``test_mse``.
    title
        Figure suptitle.
    styles
        Optional ``{label: {"color": "C0"}}`` overrides. Any label without an
        entry falls back to the default colour cycle in order.
    show_train
        If True, dashed-x lines for the train split are drawn on the variance
        and MSE panels (test is always shown solid-circle).
    save_path
        Where to write the PNG. None -> just return the Figure.

    Convention:
      - colour      distinguishes the experiment label
      - linestyle   distinguishes the split (test = solid + circle,
                                             train = dashed + x)

    Panels:
      1. Preserved variance vs k
      2. Residual entropy vs k
      3. Reconstruction MSE vs k (log y)
    """
    if not experiments:
        raise ValueError("plot_experiments_comparison requires at least one experiment")

    # Resolve a colour for every label.
    styles = dict(styles) if styles else {}
    color_iter = iter(DEFAULT_COLOR_CYCLE)
    resolved_styles = {}
    for label in experiments:
        s = dict(styles.get(label, {}))
        s.setdefault("color", next(color_iter, "black"))
        resolved_styles[label] = s

    # Pool the indices so we can set consistent log-2 ticks even if experiments
    # used different component grids.
    grid = sorted(set().union(*[df.index.tolist() for df in experiments.values()]))

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1 -- preserved variance (train + test)
    ax = axes[0]
    for label, df in experiments.items():
        c = resolved_styles[label]["color"]
        ax.plot(df.index, df["test_preserved_variance"], "-o", color=c, label=f"{label} test")
        if show_train:
            ax.plot(df.index, df["train_preserved_variance"], "--x",
                    color=c, alpha=0.6, label=f"{label} train")
    ax.set_ylabel("Preserved variance")
    ax.set_title("Preserved variance vs k")
    _style_log2_x(ax, grid)

    # Panel 2 -- residual entropy (single value per k, no train/test split)
    ax = axes[1]
    for label, df in experiments.items():
        ax.plot(df.index, df["residual_entropy"], "-o",
                color=resolved_styles[label]["color"], label=label)
    ax.set_ylabel("Residual entropy (nats, Gaussian approx.)")
    ax.set_title("Residual entropy vs k")
    _style_log2_x(ax, grid)

    # Panel 3 -- reconstruction MSE (train + test, log y)
    ax = axes[2]
    for label, df in experiments.items():
        c = resolved_styles[label]["color"]
        ax.plot(df.index, df["test_mse"], "-o", color=c, label=f"{label} test")
        if show_train:
            ax.plot(df.index, df["train_mse"], "--x",
                    color=c, alpha=0.6, label=f"{label} train")
    ax.set_yscale("log")
    ax.set_ylabel("Reconstruction MSE (log scale)")
    ax.set_title("Reconstruction MSE vs k")
    _style_log2_x(ax, grid)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    # One shared legend for the whole figure (panels 1 & 3 share it exactly;
    # panel 2 uses the same method colours), placed below so it never overlaps.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=4, fontsize=8, framealpha=0.9)
    _save(fig, save_path)
    return fig


def _style_log2_x(ax: plt.Axes, grid: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(grid)
    ax.set_xticklabels(grid)
    ax.set_xlabel("Latent / component size (k)")
    ax.grid(True, alpha=0.3, which="both")


# ============================================================================
# Downstream-prediction comparison plot (src.prediction results).
# ============================================================================
def plot_prediction_comparison(
    experiments: Mapping[str, pd.DataFrame],
    baselines: Mapping[str, Mapping[str, float]] | None = None,
    title: str = "Downstream next-step prediction",
    styles: Mapping[str, Mapping] | None = None,
    show_train: bool = True,
    save_path: str | None = None,
    figsize: tuple = (11, 4.2),
) -> plt.Figure:
    """Overlay per-method next-step prediction skill on a two-panel figure.

    Parameters
    ----------
    experiments
        ``{label: results_df}`` where each DataFrame is indexed by k and has the
        columns produced by src.prediction.run_prediction_experiment
        (``train_pred_r2`` / ``test_pred_r2`` / ``train_pred_mse`` /
        ``test_pred_mse``).
    baselines
        Optional ``{name: {"test_pred_r2": .., "test_pred_mse": ..}}`` drawn as
        horizontal reference lines (e.g. the full-X and naive baselines). These
        are constant in k.
    show_train
        Also draw the (dashed) train curves alongside the solid test curves.

    Panels: R^2 vs k (higher = better), and MSE vs k (lower = better).
    Convention: colour = method, solid+circle = test, dashed+x = train.
    """
    if not experiments:
        raise ValueError("plot_prediction_comparison requires at least one experiment")

    styles = dict(styles) if styles else {}
    color_iter = iter(DEFAULT_COLOR_CYCLE)
    resolved_styles = {}
    for label in experiments:
        s = dict(styles.get(label, {}))
        s.setdefault("color", next(color_iter, "black"))
        resolved_styles[label] = s

    grid = sorted(set().union(*[df.index.tolist() for df in experiments.values()]))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, (metric, ylabel, ttl) in zip(
        axes,
        [("r2",  "Prediction R^2",            "Next-step R^2 vs k (higher = better)"),
         ("mse", "Prediction MSE (per elem.)", "Next-step MSE vs k (lower = better)")],
    ):
        for label, df in experiments.items():
            c = resolved_styles[label]["color"]
            ax.plot(df.index, df[f"test_pred_{metric}"], "-o", color=c, label=f"{label} test")
            if show_train:
                ax.plot(df.index, df[f"train_pred_{metric}"], "--x",
                        color=c, alpha=0.6, label=f"{label} train")
        # Baselines as horizontal reference lines.
        for i, (name, vals) in enumerate((baselines or {}).items()):
            key = f"test_pred_{metric}"
            if key in vals:
                ax.axhline(vals[key], color="0.4", lw=1.1,
                           ls=(":" if i == 0 else "--"), label=f"{name} (test)")
        if metric == "r2":
            ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)  # no-skill line
        ax.set_ylabel(ylabel)
        ax.set_title(ttl)
        _style_log2_x(ax, grid)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    # One shared legend for both panels (their per-panel legends are identical).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=5, fontsize=7, framealpha=0.9)
    _save(fig, save_path)
    return fig
