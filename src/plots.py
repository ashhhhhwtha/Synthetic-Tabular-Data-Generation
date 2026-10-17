"""Three plots: column distributions, association heatmaps, nearest-neighbor distances."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import assoc_matrix


def plot_distributions(real, syn, pre, path, label="VAE"):
    cols = pre.num + pre.cat
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    for ax, c in zip(axes.ravel(), cols):
        if c in pre.num:
            lo, hi = real[c].min(), real[c].max()
            bins = np.linspace(lo, hi, 40)
            ax.hist(real[c], bins=bins, alpha=0.5, density=True, label="real")
            ax.hist(syn[c], bins=bins, alpha=0.5, density=True, label=label)
        else:
            p = real[c].value_counts(normalize=True)
            q = syn[c].value_counts(normalize=True).reindex(p.index).fillna(0)
            x = np.arange(len(p))
            ax.bar(x - 0.2, p.values, 0.4, label="real")
            ax.bar(x + 0.2, q.values, 0.4, label=label)
            ax.set_xticks(x)
            ax.set_xticklabels([str(i)[:8] for i in p.index], rotation=70, fontsize=6)
        ax.set_title(c, fontsize=9)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_assoc(real, syn, pre, path, label="VAE"):
    cols = pre.num + pre.cat
    R, S = assoc_matrix(real, pre), assoc_matrix(syn, pre)
    panels = [("real", R, -1, 1), (label, S, -1, 1), ("|real - synthetic|", np.abs(R - S), 0, 1)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (t, M, lo, hi) in zip(axes, panels):
        im = ax.imshow(M, vmin=lo, vmax=hi, cmap="RdBu_r" if lo < 0 else "viridis")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=7)
        ax.set_yticklabels(cols, fontsize=7)
        ax.set_title(t)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.text(0.5, 0.01, "Pearson (numeric pairs), Cramer's V (categorical pairs); mixed pairs blank",
             ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_nn(dists, threshold, path):
    """dists: {name: (distances, exact_share)}. Exact copies (distance 0) are excluded from bins."""
    fig, ax = plt.subplots(figsize=(8, 5))
    top = max(np.quantile(d[d > 0], 0.995) for d, _ in dists.values() if (d > 0).any())
    bins = np.linspace(0, top, 80)
    for name, (d, ex) in dists.items():
        ax.hist(d[d > 0], bins=bins, histtype="step", density=True, lw=1.5,
                label=f"{name} (exact copies: {ex:.1%})")
    ax.axvline(threshold, color="k", ls="--", label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("distance to nearest training row (preprocessed space)")
    ax.set_ylabel("density (exact copies excluded)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_length(res, path, metrics=("auc_ratio", "ks", "assoc_diff", "detect_auc",
                                     "exact_share", "median_ratio")):
    """Metric vs training epochs, one line per generator variant (mean ± std over seeds)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    epochs = sorted(res["epochs"].unique())
    for ax, m in zip(axes.ravel(), metrics):
        for name, g in res.groupby("generator", sort=False):
            a = g.groupby("epochs")[m].agg(["mean", "std"]).reindex(epochs).fillna(0)
            ax.errorbar(range(len(epochs)), a["mean"], yerr=a["std"], marker="o", capsize=3,
                        label=name)
        ax.set_xticks(range(len(epochs)))
        ax.set_xticklabels(epochs)
        ax.set_xlabel("training epochs")
        ax.set_title(m)
    axes.ravel()[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
