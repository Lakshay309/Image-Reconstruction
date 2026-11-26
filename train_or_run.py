
#!/usr/bin/env python
"""
Image Reconstruction Pipeline
Workflow:
    1. Load images
    2. Degrade images
    3. Tune parameters (optional)
    4. Reconstruct images
    5. Evaluate results
    6. Save visualizations and metrics
"""
from __future__ import annotations
import argparse
import json
import os
import time
from collections import OrderedDict
import numpy as np
import torch
from dataset import load_user_images
from degradation import degrade
from metrics import evaluate, psnr, summarize
from solver import (
    adaptive_tv,
    dip_adaptive_tv,
    dip_ablation,
    fixed_tv,
    nearest_fill,
)
from visualize import (
    plot_comparison_grid,
    plot_row_profile,
    plot_weight_map,
    save_individual,
)

# 1. COMMAND-LINE ARGUMENTS
def parse_args():
    parser = argparse.ArgumentParser(
        description="Image reconstruction using TV and Deep Image Prior"
    )
    # Image input
    image_group = parser.add_argument_group("Image input")
    image_group.add_argument(
        "--image",
        required=True,
        help="Path to an image file or directory containing images"
    )
    image_group.add_argument("--n_images", type=int, default=None)
    image_group.add_argument("--size", type=int, default=64)
    image_group.add_argument("--color", action="store_true")
    image_group.add_argument(
        "--fit",
        default="crop",
        choices=["crop", "pad"]
    )
    # Image degradation
    degradation_group = parser.add_argument_group("Image degradation")
    degradation_group.add_argument(
        "--missing",
        type=float,
        default=0.20,
        help="Fraction of pixels to remove (default: 0.20)"
    )
    degradation_group.add_argument(
        "--mask",
        default="random",
        choices=["random", "block", "mixed"]
    )
    degradation_group.add_argument(
        "--sigma",
        type=float,
        default=0.01,
        help="Gaussian noise standard deviation (default: 0.01)"
    )
    degradation_group.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Block side length in pixels (default: automatic)"
    )
    # Execution
    execution_group = parser.add_argument_group("Execution")
    execution_group.add_argument("--seed", type=int, default=0)
    execution_group.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"]
    )
    execution_group.add_argument("--out", default="results")
    execution_group.add_argument(
        "--ablation",
        action="store_true",
        help="Run all six ablation configurations on the same degraded input",
    )
    # Fixed TV
    ftv_group = parser.add_argument_group("Fixed TV")
    ftv_group.add_argument("--lam_ftv", type=float, default=0.08)
    ftv_group.add_argument("--ftv_iters", type=int, default=400)
    # Adaptive TV
    atv_group = parser.add_argument_group("Adaptive TV")
    atv_group.add_argument("--lam0", type=float, default=0.12)
    atv_group.add_argument("--kappa", type=float, default=0.05)
    atv_group.add_argument("--n_outer", type=int, default=6)
    atv_group.add_argument("--n_inner", type=int, default=150)
    atv_group.add_argument("--pilot_iters", type=int, default=300)
    atv_group.add_argument("--grad_sigma", type=float, default=0.0)
    # Deep Image Prior
    dip_group = parser.add_argument_group("Deep Image Prior")
    dip_group.add_argument("--hybrid", action="store_true")
    dip_group.add_argument("--dip_iters", type=int, default=1200)
    dip_group.add_argument("--dip_lr", type=float, default=1e-3)
    dip_group.add_argument("--dip_patience", type=int, default=120)
    dip_group.add_argument("--dip_min_iters", type=int, default=150)
    dip_group.add_argument("--dip_eval_every", type=int, default=10)
    dip_group.add_argument("--dip_val_fraction", type=float, default=0.10)
    dip_group.add_argument("--dip_reweight_every", type=int, default=100)
    dip_group.add_argument("--dip_anchor_weight", type=float, default=0.0)
    dip_group.add_argument("--dip_min_delta", type=float, default=1e-5)
    dip_group.add_argument("--dip_tv_scale", type=float, default=None)
    dip_group.add_argument("--dip_noise_std", type=float, default=0.0)
    dip_group.add_argument("--dip_channels", type=int, default=16)
    dip_group.add_argument("--dip_base", type=int, default=16)
    dip_group.add_argument("--dip_depth", type=int, default=4)
    dip_group.add_argument(
        "--ema",
        type=float,
        default=0.90,
        help="EMA coefficient for DIP output averaging (default: 0.90)"
    )
    # Validate arguments
    args = parser.parse_args()
    if not 0 <= args.missing <= 0.9:
        parser.error("--missing must be between 0 and 0.9")
    if args.sigma < 0:
        parser.error("--sigma cannot be negative")
    if args.block_size is not None and args.block_size < 2:
        parser.error("--block_size must be at least 2")
    if args.dip_lr <= 0:
        parser.error("--dip_lr must be positive")
    if not 0.0 <= args.ema < 1.0:
        parser.error("--ema must be between 0.0 (inclusive) and 1.0 (exclusive)")
    if not 0.0 < args.dip_val_fraction < 0.5:
        parser.error("--dip_val_fraction must be between 0 and 0.5")
    if args.dip_patience <= 0 or args.dip_eval_every <= 0 or args.dip_reweight_every <= 0:
        parser.error("--dip_patience, --dip_eval_every and --dip_reweight_every must be positive")
    if args.dip_min_iters < 0 or args.dip_anchor_weight < 0 or args.dip_noise_std < 0:
        parser.error("--dip_min_iters, --dip_anchor_weight and --dip_noise_std cannot be negative")
    if args.dip_anchor_weight != 0:
        parser.error("DIP-TV uses data fidelity + adaptive TV; remove the pilot anchor (--dip_anchor_weight 0)")
    if args.dip_min_delta < 0 or (args.dip_tv_scale is not None and args.dip_tv_scale < 0):
        parser.error("--dip_min_delta and --dip_tv_scale cannot be negative")
    if args.dip_channels <= 0 or args.dip_base <= 0 or args.dip_depth <= 0:
        parser.error("--dip_channels, --dip_base and --dip_depth must be positive")
    if args.size % (2 ** args.dip_depth) != 0:
        parser.error("--size must be divisible by 2**dip_depth")
    positive_args = [
        "size",
        "ftv_iters",
        "n_outer",
        "n_inner",
        "pilot_iters",
        "dip_iters",
    ]
    for name in positive_args:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.n_images is not None and args.n_images <= 0:
        parser.error("--n_images must be positive")
    return args

# 2. PARAMETER TUNING
def tune_parameters(X, Y, M, args, device):
    Y = Y.to(device)
    M = M.to(device)
    # Tune Fixed TV
    best_ftv_score = -1.0
    best_ftv_lambda = args.lam_ftv
    for lam in args.ftv_grid:
        reconstruction = fixed_tv(
            Y, M, lam, args.ftv_iters
        ).cpu()
        score = psnr(X, reconstruction).mean()
        print(
            f"[FTV] lambda={lam:.3f} | "
            f"PSNR={score:.2f}"
        )
        if score > best_ftv_score:
            best_ftv_score = score
            best_ftv_lambda = lam
    # Tune Adaptive TV
    best_atv_score = -1.0
    best_lam0 = args.lam0
    best_kappa = args.kappa
    for lam0 in args.atv_lam_grid:
        for kappa in args.atv_kappa_grid:
            reconstruction, _, _ = adaptive_tv(
                Y,
                M,
                lam0,
                kappa,
                args.n_outer,
                args.n_inner,
                args.grad_sigma,
                best_ftv_lambda,
                args.pilot_iters,
            )
            score = psnr(X, reconstruction.cpu()).mean()
            print(
                f"[ATV] lambda0={lam0:.3f} | "
                f"kappa={kappa:.3f} | "
                f"PSNR={score:.2f}"
            )
            if score > best_atv_score:
                best_atv_score = score
                best_lam0 = lam0
                best_kappa = kappa
    return best_ftv_lambda, best_lam0, best_kappa

# 3. IMAGE LOADING
def load_input_data(args):
    images, names = load_user_images(
        path=args.image,
        n=args.n_images,
        size=args.size,
        gray=not args.color,
        fit=args.fit,
    )
    if len(images) == 0:
        raise ValueError(
            f"No supported images found at: {args.image}"
        )
    return images, names

# 4. IMAGE RECONSTRUCTION
def reconstruct_images(Y, M, args, device):
    Y = Y.to(device)
    M = M.to(device)
    reconstructions = OrderedDict()
    timings = {}
    # Nearest fill
    start = time.time()
    nearest = nearest_fill(Y, M).clamp(0, 1).cpu()
    timings["Nearest fill"] = time.time() - start
    # Fixed TV
    start = time.time()
    fixed, = (fixed_tv(
        Y, M, args.lam_ftv, args.ftv_iters
    ),)
    reconstructions["Fixed TV"] = fixed.cpu()
    timings["Fixed TV"] = time.time() - start
    # Adaptive TV
    start = time.time()
    adaptive, weights, history = adaptive_tv(
        Y,
        M,
        args.lam0,
        args.kappa,
        args.n_outer,
        args.n_inner,
        args.grad_sigma,
        args.lam_ftv,
        args.pilot_iters,
    )
    reconstructions["Adaptive TV"] = adaptive.cpu()
    timings["Adaptive TV"] = time.time() - start

    # DIP-only and Fixed-TV + DIP are included in the ablation run.
    if args.ablation:
        common_dip = dict(
            n_iter=args.dip_iters,
            lr=args.dip_lr,
            ema=args.ema,
            z_ch=args.dip_channels,
            base=args.dip_base,
            depth=args.dip_depth,
            noise_std=args.dip_noise_std,
            device=device,
            seed=args.seed,
            val_fraction=args.dip_val_fraction,
            patience=args.dip_patience,
            min_iters=args.dip_min_iters,
            min_delta=args.dip_min_delta,
            eval_every=args.dip_eval_every,
            return_info=True,
        )
        start = time.time()
        dip_only, dip_only_info = dip_ablation(
            Y, M, mode="dip_only", lam=args.lam_ftv, **common_dip
        )
        reconstructions["DIP only"] = dip_only.cpu()
        timings["DIP only"] = time.time() - start

        start = time.time()
        fixed_dip, fixed_dip_info = dip_ablation(
            Y, M, mode="fixed_tv_dip", lam=args.lam_ftv, **common_dip
        )
        reconstructions["Fixed TV + DIP"] = fixed_dip.cpu()
        timings["Fixed TV + DIP"] = time.time() - start
        dip_ablation_diagnostics = {
            "DIP only": dip_only_info,
            "Fixed TV + DIP": fixed_dip_info,
        }
    else:
        dip_ablation_diagnostics = {}

    # Adaptive TV + DIP
    if args.hybrid or args.ablation:
        start = time.time()
        hybrid, dip_diagnostics = dip_adaptive_tv(
            Y,
            M,
            adaptive,
            args.lam0,
            args.kappa,
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
            anchor_weight=args.dip_anchor_weight,
            min_delta=args.dip_min_delta,
            tv_scale=args.dip_tv_scale,
            return_info=True,
        )
        reconstructions["Adaptive TV + DIP"] = hybrid.cpu()
        timings["Adaptive TV + DIP"] = time.time() - start
    else:
        dip_diagnostics = []
    if args.ablation:
        dip_diagnostics = {
            **dip_ablation_diagnostics,
            "Adaptive TV + DIP": dip_diagnostics,
        }
    return nearest, reconstructions, timings, weights, history, dip_diagnostics

# 5. EVALUATION
def evaluate_reconstructions(images, nearest, reconstructions):
    results = OrderedDict()
    results["Nearest fill"] = evaluate(images, nearest)
    for name, reconstruction in reconstructions.items():
        results[name] = evaluate(images, reconstruction)
    return results

# 6. VISUALIZATION AND SAVING
def save_results(
    images,
    observations,
    names,
    reconstructions,
    nearest,
    results,
    weights,
    args,
):
    os.makedirs(args.out, exist_ok=True)
    # Comparison grid
    plot_comparison_grid(
        images,
        observations,
        reconstructions,
        results,
        os.path.join(args.out, "comparison_grid.png"),
        n_show=min(len(images), 8),
    )
    # Individual images
    save_individual(
        os.path.join(args.out, "images"),
        names,
        images,
        observations,
        {
            "Nearest fill": nearest,
            **reconstructions,
        },
    )
    # Adaptive weight map
    plot_weight_map(
        reconstructions["Adaptive TV"][0],
        weights[0],
        os.path.join(args.out, "lambda_map.png"),
    )
    # Row profile
    row_energy = torch.abs(
        images[0, :, :, 1:] - images[0, :, :, :-1]
    ).sum(dim=(0, 2))
    row = int(row_energy.argmax())
    plot_row_profile(
        images[0],
        {
            name: reconstruction[0]
            for name, reconstruction in reconstructions.items()
        },
        row,
        os.path.join(args.out, "row_profile.png"),
    )

# 7. MAIN APPLICATION
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    os.makedirs(args.out, exist_ok=True)
    print(f"Device: {device}")
    if args.hybrid:
        print(
            "DIP model: "
            f"base={args.dip_base}, depth={args.dip_depth}, "
            f"input_channels={args.dip_channels}, "
            f"generator=DIP skip U-Net (per-image optimization; no pretrained weights)"
        )
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"CUDA/PyTorch: {torch.version.cuda}")
    print(f"Input: {args.image}")
    if args.ablation:
        print("Ablation mode: Nearest Fill, Fixed TV, Adaptive TV, DIP only, "
              "Fixed TV + DIP, Adaptive TV + DIP")
    # Load user images
    images, names = load_input_data(args)
    print(f"Loaded {len(images)} image(s)")
    print(f"Image shape: {tuple(images.shape[1:])}")
    # Degrade images
    observations, mask = degrade(
        images,
        missing_ratio=args.missing,
        sigma=args.sigma,
        kind=args.mask,
        seed=args.seed,
        block_size=args.block_size,
    )
    actual_missing = 1.0 - mask.mean().item()
    print(f"\nDegradation settings:")
    print(f"Missing ratio: {actual_missing:.2%}")
    print(f"Noise sigma: {args.sigma}")
    print(f"Mask type: {args.mask}")
    # Reconstruct
    (
        nearest,
        reconstructions,
        timings,
        weights,
        history,
        dip_diagnostics,
    ) = reconstruct_images(
        observations,
        mask,
        args,
        device,
    )
    print("\nAdaptive TV objective history:")
    print(" -> ".join(f"{value:.3f}" for value in history))
    # Evaluate
    results = evaluate_reconstructions(
        images,
        nearest,
        reconstructions,
    )
    print("\n" + summarize(results, timings))
    if dip_diagnostics:
        print("\nDIP ablation diagnostics (baseline comparisons do not replace DIP outputs):")
        for method, entries in dip_diagnostics.items():
            if isinstance(entries, list):
                for index, entry in enumerate(entries):
                    flag = entry.get("dip_beats_baseline", entry.get("dip_beats_training_pilot"))
                    print(
                        f"  {method} [image {index + 1}]: "
                        f"selected={entry.get('selected_method', 'unknown')}, "
                        f"beats_baseline={flag}, "
                        f"best_iteration={entry.get('best_iteration')}, "
                        f"validation_mse={entry.get('best_validation_mse')}"
                    )
            elif isinstance(entries, dict):
                flag = entries.get("dip_beats_baseline", entries.get("dip_beats_training_pilot"))
                print(
                    f"  {method}: selected={entries.get('selected_method', 'unknown')}, "
                    f"beats_baseline={flag}, "
                    f"best_iteration={entries.get('best_iteration')}, "
                    f"validation_mse={entries.get('best_validation_mse')}"
                )
    # Save figures
    save_results(
        images,
        observations,
        names,
        reconstructions,
        nearest,
        results,
        weights,
        args,
    )
    # Save metrics
    summary = {
        "arguments": vars(args),
        "degradation": {
            "requested_missing_ratio": args.missing,
            "actual_missing_ratio": actual_missing,
            "noise_sigma": args.sigma,
            "mask_type": args.mask,
            "block_size": args.block_size,
        },
        "metrics": {
            method: {
                metric: {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                }
                for metric, values in scores.items()
            }
            for method, scores in results.items()
        },
        "per_image": {
            name: {
                method: {
                    metric: float(values[index])
                    for metric, values in scores.items()
                }
                for method, scores in results.items()
            }
            for index, name in enumerate(names)
        },
        "timings": timings,
        "adaptive_tv_history": history,
        "dip_diagnostics": dip_diagnostics,
    }
    with open(
        os.path.join(args.out, "metrics.json"),
        "w",
    ) as file:
        json.dump(summary, file, indent=4)
    print(f"\nResults saved in: {args.out}")

if __name__ == "__main__":
    main()
