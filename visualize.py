"""visualize.py -- matplotlib figures: comparison grid, lambda(i,j) map, 1-D intensity profile."""
from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _show(ax, img, **kw):
    a = img.detach().cpu().clamp(0, 1).numpy()
    if a.shape[0] == 1:
        ax.imshow(a[0], cmap="gray", vmin=0, vmax=1, **kw)
    else:
        ax.imshow(a.transpose(1, 2, 0), **kw)
    ax.set_xticks([])
    ax.set_yticks([])


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_grid(x, y, recon: dict, per_image: dict | None = None,
                        path: str = "results/comparison_grid.png", n_show: int = 4):
    """Rows = images. Columns = Original | Corrupted (y) | one column per method in `recon`.
    recon: {"Fixed TV": tensor(B,C,H,W), "Adaptive TV": ...}; per_image: {name: {"psnr":arr,"ssim":arr}}."""
    n = min(n_show, x.shape[0])
    names = list(recon)
    titles = ["Original", "Corrupted ($y$)"] + names
    fig, axes = plt.subplots(n, len(titles), figsize=(2.4 * len(titles), 2.5 * n), squeeze=False)
    for i in range(n):
        imgs = [x[i], y[i]] + [recon[k][i] for k in names]
        for j, im in enumerate(imgs):
            _show(axes[i, j], im)
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=10)
            if per_image is not None and j >= 2:
                m = per_image[names[j - 2]]
                axes[i, j].set_xlabel(f"PSNR {m['psnr'][i]:.2f} dB | SSIM {m['ssim'][i]:.3f}", fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_weight_map(x_hat, W, path: str = "results/lambda_map.png", title_img: str = "Adaptive TV result"):
    """Shows the spatially varying weight lambda(i,j): bright = strong smoothing, dark = edges."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.3))
    _show(axes[0], x_hat)
    axes[0].set_title(title_img, fontsize=10)
    im = axes[1].imshow(W[0].detach().cpu().numpy(), cmap="magma")
    axes[1].set_title(r"weight map $\lambda(i,j)$", fontsize=10)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    _save(fig, path)


def plot_row_profile(x, recon: dict, row: int, path: str = "results/row_profile.png"):
    """Intensity along one image row: reveals edge blunting (contrast loss) and staircasing."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(x.mean(0)[row].cpu().numpy(), "k-", lw=2, label="Original")
    for name, r in recon.items():
        ax.plot(r.mean(0)[row].cpu().numpy(), lw=1.2, label=name)
    ax.set_title(f"Intensity profile, row {row}", fontsize=10)
    ax.set_xlabel("column")
    ax.set_ylabel("intensity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def save_individual(out_dir: str, names: list, x, y, recon: dict):
    """Save every image of every method as its own PNG:
    <out_dir>/<image_name>/{original, corrupted, <method>}.png  (corrupted: missing pixels = black)."""
    for i, nm in enumerate(names):
        d = os.path.join(out_dir, nm)
        os.makedirs(d, exist_ok=True)
        items = {"original": x[i], "corrupted": y[i]}
        items.update({re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_"): v[i] for k, v in recon.items()})
        for key, t in items.items():
            a = t.detach().cpu().clamp(0, 1).numpy()
            if a.shape[0] == 1:
                plt.imsave(os.path.join(d, key + ".png"), a[0], cmap="gray", vmin=0, vmax=1)
            else:
                plt.imsave(os.path.join(d, key + ".png"), a.transpose(1, 2, 0))

def plot_metric_summary(image_level_frame, out_dir: str):
    """Generate per-image distribution plots for PSNR, SSIM and Edge-PSNR."""
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    metrics = [
        ("psnr", "PSNR (dB)", "psnr_comparison.png"),
        ("ssim", "SSIM", "ssim_comparison.png"),
        ("edge_psnr", "Edge-PSNR (dB)", "edge_psnr_comparison.png"),
    ]
    for metric, label, filename in metrics:
        methods = list(image_level_frame["method"].drop_duplicates())
        data = [
            image_level_frame.loc[image_level_frame.method == method, metric].dropna().to_numpy()
            for method in methods
        ]
        fig, ax = plt.subplots(figsize=(max(9, len(methods) * 1.5), 5))
        # Matplotlib renamed the `labels` argument to `tick_labels`.
        # Support both newer and older Matplotlib releases.
        try:
            ax.boxplot(data, tick_labels=methods, showmeans=True)
        except TypeError as exc:
            if "tick_labels" not in str(exc):
                raise
            ax.boxplot(data, labels=methods, showmeans=True)
        ax.set_ylabel(label)
        ax.set_title(f"{label} across reference images (seed-averaged per image)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, filename))

    # Mean with standard-deviation error bars across independent images.
    for metric, label, filename in metrics:
        grouped = image_level_frame.groupby("method")[metric]
        means, stds = grouped.mean(), grouped.std(ddof=1).fillna(0)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(means.index, means.values, yerr=stds.values, capsize=4)
        ax.set_ylabel(label)
        ax.set_title(f"Mean {label} ± SD across images")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, filename.replace(".png", "_mean_sd.png")))


