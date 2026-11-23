#!/usr/bin/env python
"""Standalone lambda0/kappa sensitivity study for Adaptive TV + DIP.

This module is deliberately separate from train_or_run.py. It creates one
observation/mask, reuses them for every grid point, uses the same validation
split and DIP initialization seed for every candidate, and selects ONLY by
held-out observation validation MSE. Ground-truth metrics are calculated only
once, after the winning configuration has been selected.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import load_user_images
from degradation import degrade
from metrics import evaluate
from solver import adaptive_tv, dip_adaptive_tv


def _float_list(value: str) -> list[float]:
    try:
        values = [float(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers") from exc
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise argparse.ArgumentTypeError("All grid values must be finite and > 0")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Grid values must not contain duplicates")
    return values


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare Adaptive TV + DIP over a lambda0/kappa grid."
    )
    p.add_argument("--image", required=True, help="Image path or directory")
    p.add_argument("--n_images", type=int, default=None)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--color", action="store_true")
    p.add_argument("--fit", choices=["crop", "pad"], default="crop")

    p.add_argument("--missing", type=float, default=0.10)
    p.add_argument("--mask", choices=["random", "block", "mixed"], default="block")
    p.add_argument("--sigma", type=float, default=0.08)
    p.add_argument("--block_size", type=int, default=8)

    p.add_argument("--seed", type=int, default=0,
                   help="Shared degradation, validation-split and DIP seed")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out", default="hyperparam_results")

    p.add_argument("--lambda0_grid", type=_float_list,
                   default=_float_list("0.08,0.10,0.12,0.14,0.16"))
    p.add_argument("--kappa_grid", type=_float_list,
                   default=_float_list("0.025,0.05,0.075,0.10"))

    # Adaptive-TV pilot/solver settings
    p.add_argument("--n_outer", type=int, default=6)
    p.add_argument("--n_inner", type=int, default=150)
    p.add_argument("--pilot_iters", type=int, default=300)
    p.add_argument("--grad_sigma", type=float, default=0.0)

    # DIP settings (kept aligned with the normal experiment defaults)
    p.add_argument("--dip_iters", type=int, default=1200)
    p.add_argument("--dip_lr", type=float, default=1e-3)
    p.add_argument("--dip_patience", type=int, default=120)
    p.add_argument("--dip_min_iters", type=int, default=100)
    p.add_argument("--dip_eval_every", type=int, default=10)
    p.add_argument("--dip_val_fraction", type=float, default=0.10)
    p.add_argument("--dip_reweight_every", type=int, default=100)
    p.add_argument("--dip_min_delta", type=float, default=1e-5)
    p.add_argument("--dip_tv_scale", type=float, default=None)
    p.add_argument("--dip_noise_std", type=float, default=0.0)
    p.add_argument("--dip_channels", type=int, default=16)
    p.add_argument("--dip_base", type=int, default=16)
    p.add_argument("--dip_depth", type=int, default=4)
    p.add_argument("--ema", type=float, default=0.90)

    a = p.parse_args()
    if not 0 <= a.missing <= 0.9:
        p.error("--missing must be between 0 and 0.9")
    if a.sigma < 0 or a.dip_noise_std < 0:
        p.error("Noise standard deviations cannot be negative")
    if a.block_size is not None and a.block_size < 2:
        p.error("--block_size must be at least 2")
    if a.dip_lr <= 0 or not 0 <= a.ema < 1:
        p.error("--dip_lr must be positive and --ema must be in [0,1)")
    if not 0 < a.dip_val_fraction < 0.5:
        p.error("--dip_val_fraction must be between 0 and 0.5")
    if a.dip_patience <= 0 or a.dip_eval_every <= 0 or a.dip_reweight_every <= 0:
        p.error("DIP patience/eval/reweight intervals must be positive")
    if a.dip_min_iters < 0 or a.dip_min_delta < 0:
        p.error("DIP minimum iterations/delta cannot be negative")
    if a.dip_tv_scale is not None and a.dip_tv_scale < 0:
        p.error("--dip_tv_scale cannot be negative")
    if min(a.n_outer, a.n_inner, a.pilot_iters, a.dip_iters,
           a.dip_channels, a.dip_base, a.dip_depth, a.size) <= 0:
        p.error("Solver sizes/iterations/channels/depth must be positive")
    if a.size % (2 ** a.dip_depth) != 0:
        p.error("--size must be divisible by 2**dip_depth")
    return a


def _save_tensor_image(tensor: torch.Tensor, path: Path):
    arr = tensor.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    arr = np.round(arr * 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        Image.fromarray(arr[..., 0], mode="L").save(path)
    else:
        Image.fromarray(arr[..., :3], mode="RGB").save(path)


def _json_safe(value):
    """Replace NaN/Infinity with null so strict JSON readers accept output."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _write_csv(path: Path, rows: list[dict]):
    fields = [
        "lambda0", "kappa", "status", "selected", "mean_validation_mse",
        "mean_training_pilot_validation_mse", "images_selected", "image_count",
        "runtime_seconds", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _save_heatmap(path: Path, lambdas, kappas, values, title, colorbar_label):
    data = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(kappas)), [f"{v:g}" for v in kappas])
    ax.set_yticks(range(len(lambdas)), [f"{v:g}" for v in lambdas])
    ax.set_xlabel("kappa")
    ax.set_ylabel("lambda0")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(colorbar_label)
    for i in range(len(lambdas)):
        for j in range(len(kappas)):
            v = data[i, j]
            ax.text(j, i, "N/A" if not np.isfinite(v) else f"{v:.5g}",
                    ha="center", va="center", fontsize=8,
                    color="white" if np.isfinite(v) else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print("=" * 72)
    print("ADAPTIVE TV + DIP HYPERPARAMETER COMPARISON")
    print("=" * 72)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"lambda0 grid: {args.lambda0_grid}")
    print(f"kappa grid:   {args.kappa_grid}")
    print(f"Total configurations: {len(args.lambda0_grid) * len(args.kappa_grid)}")
    print("Selection criterion: held-out observed-pixel validation MSE (lower is better)")
    print("Ground-truth metrics are NOT used to select hyperparameters.")

    images, names = load_user_images(
        path=args.image, n=args.n_images, size=args.size,
        gray=not args.color, fit=args.fit,
    )
    if len(images) == 0:
        raise ValueError(f"No supported images found at: {args.image}")
    observations, mask = degrade(
        images, missing_ratio=args.missing, sigma=args.sigma,
        kind=args.mask, seed=args.seed, block_size=args.block_size,
    )
    y, m = observations.to(device), mask.to(device)
    actual_missing = float(1.0 - mask.mean().item())
    print(f"Loaded {len(images)} image(s), shape={tuple(images.shape[1:])}")
    print(f"One shared degradation generated: missing={actual_missing:.4%}, sigma={args.sigma}, mask={args.mask}")

    rows: list[dict] = []
    diagnostics_by_config = {}
    reconstructions_by_config = {}
    total = len(args.lambda0_grid) * len(args.kappa_grid)
    config_index = 0

    for lam0 in args.lambda0_grid:
        for kappa in args.kappa_grid:
            config_index += 1
            label = f"lambda0={lam0:g}, kappa={kappa:g}"
            print(f"\n[{config_index}/{total}] {label}", flush=True)
            started = time.perf_counter()
            try:
                # The ATV pilot is specific to this pair and uses the exact same y/M.
                pilot, _, _ = adaptive_tv(
                    y, m, lam0, kappa,
                    n_outer=args.n_outer, n_inner=args.n_inner,
                    grad_sigma=args.grad_sigma,
                    pilot_lam=lam0, pilot_iters=args.pilot_iters,
                )
                # dip_adaptive_tv derives its held-out split from seed+7919.
                # Passing the same seed each time guarantees the same split and
                # same fresh DIP initialization across all grid configurations.
                result, diagnostics = dip_adaptive_tv(
                    y, m, pilot, lam0, kappa,
                    n_iter=args.dip_iters,
                    lr=args.dip_lr,
                    reweight_every=args.dip_reweight_every,
                    ema=args.ema,
                    grad_sigma=args.grad_sigma,
                    z_ch=args.dip_channels,
                    base=args.dip_base,
                    depth=args.dip_depth,
                    noise_std=args.dip_noise_std,
                    device=device,
                    seed=args.seed,
                    val_fraction=args.dip_val_fraction,
                    patience=args.dip_patience,
                    min_iters=args.dip_min_iters,
                    eval_every=args.dip_eval_every,
                    min_delta=args.dip_min_delta,
                    tv_scale=args.dip_tv_scale,
                    return_info=True,
                )
                elapsed = time.perf_counter() - started
                selected_flags = [bool(d.get("dip_selected_on_validation", False)) for d in diagnostics]
                val_scores = [float(d.get("best_validation_mse", float("inf"))) for d in diagnostics]
                pilot_scores = [float(d.get("training_pilot_validation_mse", float("inf"))) for d in diagnostics]
                all_selected = bool(selected_flags) and all(selected_flags)
                finite_scores = [v for v in val_scores if math.isfinite(v)]
                mean_val = float(np.mean(finite_scores)) if len(finite_scores) == len(val_scores) and finite_scores else float("inf")
                mean_pilot = float(np.mean(pilot_scores)) if pilot_scores and all(math.isfinite(v) for v in pilot_scores) else float("inf")
                row = {
                    "lambda0": lam0, "kappa": kappa,
                    "status": "completed" if all_selected else "DIP_not_selected_for_all_images",
                    "selected": False,
                    "mean_validation_mse": mean_val,
                    "mean_training_pilot_validation_mse": mean_pilot,
                    "images_selected": sum(selected_flags),
                    "image_count": len(images),
                    "runtime_seconds": round(elapsed, 3),
                    "error": "",
                }
                diagnostics_by_config[f"{lam0:g}|{kappa:g}"] = diagnostics
                reconstructions_by_config[f"{lam0:g}|{kappa:g}"] = result.detach().cpu()
                print(
                    f"  val MSE={mean_val:.8f} | pilot val MSE={mean_pilot:.8f} | "
                    f"DIP selected={sum(selected_flags)}/{len(images)} | {elapsed:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                row = {
                    "lambda0": lam0, "kappa": kappa, "status": "failed",
                    "selected": False, "mean_validation_mse": float("inf"),
                    "mean_training_pilot_validation_mse": float("inf"),
                    "images_selected": 0, "image_count": len(images),
                    "runtime_seconds": round(elapsed, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  FAILED after {elapsed:.1f}s: {row['error']}", flush=True)
                with (out_dir / "errors.log").open("a", encoding="utf-8") as ef:
                    ef.write(f"\n--- {label} ---\n{traceback.format_exc()}\n")
            rows.append(row)
            _write_csv(out_dir / "hyperparameter_results.csv", rows)
            # Persist progress after every pair so an interruption does not erase completed runs.
            with (out_dir / "progress.json").open("w", encoding="utf-8") as pf:
                json.dump({"completed": config_index, "total": total, "rows": rows}, pf, indent=2)

    eligible = [r for r in rows if r["status"] == "completed" and math.isfinite(r["mean_validation_mse"])]
    if not eligible:
        raise RuntimeError(
            "No configuration produced a DIP candidate that beat the training-only ATV pilot "
            "for every image. Check errors.log and the per-configuration diagnostics."
        )
    winner = min(eligible, key=lambda r: r["mean_validation_mse"])
    winner["selected"] = True
    best_lam, best_kappa = float(winner["lambda0"]), float(winner["kappa"])
    best_key = f"{best_lam:g}|{best_kappa:g}"
    best_reconstruction = reconstructions_by_config[best_key]
    best_diagnostics = diagnostics_by_config[best_key]
    _write_csv(out_dir / "hyperparameter_results.csv", rows)

    # Grid matrices for the report; invalid/fallback configurations remain NaN.
    val_matrix = np.full((len(args.lambda0_grid), len(args.kappa_grid)), np.nan)
    pilot_matrix = np.full_like(val_matrix, np.nan)
    for row in rows:
        if row["status"] == "completed":
            i = args.lambda0_grid.index(float(row["lambda0"]))
            j = args.kappa_grid.index(float(row["kappa"]))
            val_matrix[i, j] = row["mean_validation_mse"]
            pilot_matrix[i, j] = row["mean_training_pilot_validation_mse"]
    _save_heatmap(out_dir / "validation_mse_heatmap.png", args.lambda0_grid,
                  args.kappa_grid, val_matrix,
                  "Adaptive TV + DIP: held-out validation MSE", "Validation MSE (lower is better)")
    _save_heatmap(out_dir / "training_pilot_validation_heatmap.png", args.lambda0_grid,
                  args.kappa_grid, pilot_matrix,
                  "Adaptive TV pilot: held-out validation MSE", "Pilot validation MSE (lower is better)")

    # Only now, after selecting by held-out validation, inspect ground-truth metrics.
    gt_metrics = evaluate(images, best_reconstruction)
    metric_summary = {
        key: {"mean": float(values.mean()), "std": float(values.std()),
              "per_image": [float(v) for v in values]}
        for key, values in gt_metrics.items()
    }
    _save_tensor_image(best_reconstruction, out_dir / "best_reconstruction.png")
    _save_tensor_image(images, out_dir / "original.png")
    _save_tensor_image(observations, out_dir / "degraded_observation.png")

    best_config = {
        "selection_rule": "Lowest mean held-out observed-pixel validation MSE among configurations where DIP beat the training-only Adaptive TV pilot for every image.",
        "selected_parameters": {"lambda0": best_lam, "kappa": best_kappa},
        "selection_validation_mse": winner["mean_validation_mse"],
        "training_pilot_validation_mse": winner["mean_training_pilot_validation_mse"],
        "ground_truth_metrics_after_selection": metric_summary,
        "image_names": names,
        "experiment": {
            "image": args.image, "size": args.size, "color": args.color,
            "fit": args.fit, "seed": args.seed, "device": str(device),
            "requested_missing_ratio": args.missing,
            "actual_missing_ratio": actual_missing, "mask": args.mask,
            "sigma": args.sigma, "block_size": args.block_size,
            "n_outer": args.n_outer, "n_inner": args.n_inner,
            "pilot_iters": args.pilot_iters, "grad_sigma": args.grad_sigma,
            "dip_iters": args.dip_iters, "dip_lr": args.dip_lr,
            "dip_patience": args.dip_patience, "dip_min_iters": args.dip_min_iters,
            "dip_eval_every": args.dip_eval_every,
            "dip_val_fraction": args.dip_val_fraction,
            "dip_reweight_every": args.dip_reweight_every,
            "dip_min_delta": args.dip_min_delta,
            "dip_tv_scale": args.dip_tv_scale,
            "dip_noise_std": args.dip_noise_std,
            "dip_channels": args.dip_channels, "dip_base": args.dip_base,
            "dip_depth": args.dip_depth, "ema": args.ema,
            "lambda0_grid": args.lambda0_grid, "kappa_grid": args.kappa_grid,
        },
        "selected_dip_diagnostics": best_diagnostics,
    }
    with (out_dir / "best_configuration.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(best_config), f, indent=2, allow_nan=False)
    with (out_dir / "all_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics_by_config, f, indent=2, allow_nan=True)

    print("\n" + "=" * 72)
    print("HYPERPARAMETER STUDY COMPLETE")
    print("=" * 72)
    print(f"Best lambda0: {best_lam:g}")
    print(f"Best kappa:   {best_kappa:g}")
    print(f"Selection validation MSE: {winner['mean_validation_mse']:.8f}")
    print("Ground-truth metrics (computed only after selection):")
    for metric, values in gt_metrics.items():
        print(f"  {metric}: {values.mean():.6f} ± {values.std():.6f}")
    print(f"\nAll outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
