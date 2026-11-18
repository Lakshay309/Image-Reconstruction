"""metrics.py -- PSNR, SSIM and an edge-focused PSNR (all images in [0,1], tensors (B,C,H,W))."""
from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _np(t: torch.Tensor) -> np.ndarray:
    """(B,C,H,W) tensor -> (B,H,W,C) float64 array."""
    return t.detach().cpu().double().numpy().transpose(0, 2, 3, 1)


def psnr(x: torch.Tensor, xhat: torch.Tensor) -> np.ndarray:
    """Per-image PSNR in dB (peak = 1)."""
    return np.array([peak_signal_noise_ratio(a, b, data_range=1.0) for a, b in zip(_np(x), _np(xhat))])


def ssim(x: torch.Tensor, xhat: torch.Tensor) -> np.ndarray:
    """Per-image mean SSIM (7x7 uniform window, skimage default)."""
    out = []
    for a, b in zip(_np(x), _np(xhat)):
        if a.shape[-1] == 1:
            out.append(structural_similarity(a[..., 0], b[..., 0], data_range=1.0))
        else:
            out.append(structural_similarity(a, b, data_range=1.0, channel_axis=-1))
    return np.array(out)


def edge_psnr(x: torch.Tensor, xhat: torch.Tensor, q: float = 0.9) -> np.ndarray:
    """PSNR restricted to the top-(1-q) fraction of pixels by ground-truth gradient magnitude.
    Directly measures how well edges / fine structure survive (TV-type bias shows up here)."""
    out = []
    for a, b in zip(_np(x), _np(xhat)):
        gy, gx = np.gradient(a, axis=(0, 1))
        mag = np.sqrt((gx ** 2 + gy ** 2).sum(-1))
        mask = mag >= np.quantile(mag, q)
        mse = np.mean((a[mask] - b[mask]) ** 2)
        out.append(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    return np.array(out)


def evaluate(x: torch.Tensor, xhat: torch.Tensor) -> dict:
    return {"psnr": psnr(x, xhat), "ssim": ssim(x, xhat), "edge_psnr": edge_psnr(x, xhat)}


def summarize(results: dict, times: dict | None = None) -> str:
    """results: {method: evaluate(...) dict}. Returns a printable table (mean +- std over images)."""
    head = f"{'method':<22}{'PSNR (dB)':>16}{'SSIM':>16}{'edge-PSNR (dB)':>20}" + (f"{'time (s)':>11}" if times else "")
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        row = (f"{name:<22}{r['psnr'].mean():>9.2f} ±{r['psnr'].std():>4.2f}"
               f"{r['ssim'].mean():>10.4f} ±{r['ssim'].std():>5.3f}"
               f"{r['edge_psnr'].mean():>13.2f} ±{r['edge_psnr'].std():>4.2f}")
        if times:
            row += f"{times.get(name, float('nan')):>11.1f}"
        lines.append(row)
    return "\n".join(lines)
