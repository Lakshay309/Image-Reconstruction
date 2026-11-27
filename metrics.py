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

def descriptive_statistics(values, seed: int = 0, n_boot: int = 10000) -> dict:
    """Descriptive statistics and percentile-bootstrap 95% CI for the mean."""
    from scipy import stats
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {k: None for k in (
            "mean", "median", "mode", "min", "max", "std", "variance",
            "q1", "q3", "iqr", "ci95_low", "ci95_high"
        )}
    mode_value = None
    if a.size > 1:
        vals, counts = np.unique(a, return_counts=True)
        if counts.max() > 1:
            mode_value = float(vals[counts == counts.max()][0])
    rng = np.random.default_rng(seed)
    if a.size == 1:
        ci_low = ci_high = float(a[0])
    else:
        samples = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
        ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "mode": mode_value,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "variance": float(np.var(a, ddof=1)) if a.size > 1 else 0.0,
        "q1": float(np.quantile(a, 0.25)),
        "q3": float(np.quantile(a, 0.75)),
        "iqr": float(np.quantile(a, 0.75) - np.quantile(a, 0.25)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def _holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def paired_statistical_tests(image_level_frame, metrics=("psnr", "ssim", "edge_psnr")):
    """Paired t/Wilcoxon tests across image-level means, with Holm correction."""
    from itertools import combinations
    from scipy import stats
    methods = sorted(image_level_frame["method"].unique())
    output = []
    for metric in metrics:
        pending = []
        for a_name, b_name in combinations(methods, 2):
            a = image_level_frame.loc[image_level_frame.method == a_name].set_index("image")[metric]
            b = image_level_frame.loc[image_level_frame.method == b_name].set_index("image")[metric]
            joined = np.isfinite(a) & np.isfinite(b)
            a_vals, b_vals = a[joined].to_numpy(float), b[joined].to_numpy(float)
            diff = a_vals - b_vals
            n = len(diff)
            if n < 2:
                t_stat = t_p = w_stat = w_p = dz = mean_diff = np.nan
            else:
                mean_diff = float(np.mean(diff))
                sd = float(np.std(diff, ddof=1))
                dz = mean_diff / sd if sd > 0 else (0.0 if mean_diff == 0 else float("inf"))
                t_res = stats.ttest_rel(a_vals, b_vals, nan_policy="omit")
                t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
                try:
                    w_res = stats.wilcoxon(a_vals, b_vals, zero_method="wilcox", alternative="two-sided")
                    w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
                except ValueError:
                    w_stat, w_p = 0.0, 1.0
            pending.append({
                "metric": metric, "method_a": a_name, "method_b": b_name,
                "n_pairs": n, "paired_t": t_stat, "paired_t_p": t_p,
                "wilcoxon_w": w_stat, "wilcoxon_p": w_p,
                "cohens_dz": dz, "mean_difference": mean_diff,
            })
        for p_col, adjusted_col in (
            ("paired_t_p", "holm_adjusted_p_t"),
            ("wilcoxon_p", "holm_adjusted_p_wilcoxon"),
        ):
            valid_idx = [i for i, row in enumerate(pending) if np.isfinite(row[p_col])]
            adjusted = _holm_adjust([pending[i][p_col] for i in valid_idx]) if valid_idx else []
            for i, value in zip(valid_idx, adjusted):
                pending[i][adjusted_col] = float(value)
            for i, row in enumerate(pending):
                row.setdefault(adjusted_col, np.nan)
        output.extend(pending)
    # Friedman omnibus test across all six methods, using complete image blocks.
    for metric in metrics:
        wide = image_level_frame.pivot(index="image", columns="method", values=metric)
        wide = wide.dropna(axis=0, how="any")
        if wide.shape[0] >= 2 and wide.shape[1] >= 3:
            try:
                fr = stats.friedmanchisquare(*[wide[col].to_numpy(float) for col in wide.columns])
                output.append({
                    "metric": metric, "method_a": "ALL_METHODS",
                    "method_b": "Friedman omnibus",
                    "n_pairs": int(wide.shape[0]),
                    "paired_t": np.nan, "paired_t_p": np.nan,
                    "wilcoxon_w": float(fr.statistic), "wilcoxon_p": float(fr.pvalue),
                    "cohens_dz": np.nan, "mean_difference": np.nan,
                    "holm_adjusted_p_t": np.nan,
                    "holm_adjusted_p_wilcoxon": np.nan,
                })
            except ValueError:
                pass
    return output



