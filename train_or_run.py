
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
    dip_group.add_argument("--dip_iters", type=int, default=800)
    dip_group.add_argument("--dip_lr", type=float, default=1e-2)
    dip_group.add_argument(
        "--ema",
        type=float,
        default=0.95,
        help="EMA coefficient for DIP output averaging (default: 0.95)"
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
    # Adaptive TV + DIP
    if args.hybrid:
        start = time.time()
        hybrid = dip_adaptive_tv(
            Y,
            M,
            adaptive,
            args.lam0,
            args.kappa,
            n_iter=args.dip_iters,
            lr=args.dip_lr,
            ema=args.ema,
            grad_sigma=args.grad_sigma,
            device=device,
            seed=args.seed,
        )
        reconstructions["Adaptive TV + DIP"] = hybrid.cpu()
        timings["Adaptive TV + DIP"] = time.time() - start
    return nearest, reconstructions, timings, weights, history

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
    print(f"Input: {args.image}")
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
    }
    with open(
        os.path.join(args.out, "metrics.json"),
        "w",
    ) as file:
        json.dump(summary, file, indent=4)
    print(f"\nResults saved in: {args.out}")

if __name__ == "__main__":
    main()
