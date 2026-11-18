#!/usr/bin/env python
"""train_or_run.py -- end-to-end experiment: load -> degrade -> reconstruct -> evaluate -> plot.

Methods compared
    Nearest fill        (reference: copy nearest observed pixel)
    Fixed TV            (baseline, lambda grid-tuned if --tune)
    Adaptive TV         (iteratively reweighted TV / MM; lam0, kappa grid-tuned if --tune)
    Adaptive TV + DIP   (only with --hybrid; U-Net prior + adaptive-TV penalty)

Examples
    python train_or_run.py --dataset skimage --n_images 8 --tune
    python train_or_run.py --image my_photo.jpg --size 128 --missing 0.5 --tune
    python train_or_run.py --image D:/pics/ --n_images 20 --fit pad --hybrid
    python train_or_run.py --dataset stl10 --missing 0.6 --mask mixed --tune --hybrid
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict

import numpy as np
import torch

from dataset import load_images, load_user_images
from degradation import degrade
from metrics import evaluate, psnr, summarize
from solver import (adaptive_tv, adaptive_weights, dip_adaptive_tv, fixed_tv, nearest_fill)
from visualize import plot_comparison_grid, plot_row_profile, plot_weight_map, save_individual


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive TV image recovery from incomplete information")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "stl10", "skimage"])
    p.add_argument("--data_root", default="./data")
    p.add_argument("--image", default=None,
                    help="path to YOUR image file or a folder of images (overrides --dataset)")
    p.add_argument("--fit", default="crop", choices=["crop", "pad"],
                    help="make user images square by centre-cropping or edge-padding")
    p.add_argument("--n_images", type=int, default=8)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--color", action="store_true", help="use RGB (default: grayscale)")
    p.add_argument("--missing", type=float, default=0.5, help="fraction of missing pixels")
    p.add_argument("--mask", default="random", choices=["random", "block", "mixed"])
    p.add_argument("--block_size", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.05, help="noise std on observed pixels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    # FTV
    p.add_argument("--lam_ftv", type=float, default=0.08)
    p.add_argument("--ftv_iters", type=int, default=400)
    # ATV
    p.add_argument("--lam0", type=float, default=0.12)
    p.add_argument("--kappa", type=float, default=0.05)
    p.add_argument("--n_outer", type=int, default=6)
    p.add_argument("--n_inner", type=int, default=150)
    p.add_argument("--pilot_iters", type=int, default=300)
    p.add_argument("--grad_sigma", type=float, default=0.0, help="Gaussian pre-smoothing of the pilot gradient")
    # tuning (on held-out images, so the baseline is as strong as possible)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--n_tune", type=int, default=4)
    p.add_argument("--ftv_grid", type=float, nargs="+", default=[0.02, 0.04, 0.06, 0.08, 0.12, 0.16])
    p.add_argument("--atv_lam_grid", type=float, nargs="+", default=[0.06, 0.10, 0.15, 0.22])
    p.add_argument("--atv_kappa_grid", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    # hybrid
    p.add_argument("--hybrid", action="store_true", help="also run Adaptive TV + Deep Image Prior")
    p.add_argument("--dip_iters", type=int, default=800)
    p.add_argument("--dip_lr", type=float, default=1e-2)
    p.add_argument("--out", default="results")
    return p.parse_args()


def tune(Xt, Yt, Mt, a, dev):
    """Grid-search FTV lambda, then (lam0, kappa) for ATV, maximising mean PSNR on held-out images."""
    Yt, Mt = Yt.to(dev), Mt.to(dev)
    best_f = (-1.0, a.lam_ftv)
    for lam in a.ftv_grid:
        s = psnr(Xt, fixed_tv(Yt, Mt, lam, a.ftv_iters).cpu()).mean()
        print(f"  [tune] FTV lam={lam:.3f}                PSNR={s:.2f}")
        if s > best_f[0]:
            best_f = (s, lam)
    best_a = (-1.0, a.lam0, a.kappa)
    for lam0 in a.atv_lam_grid:
        for kap in a.atv_kappa_grid:
            xh, _, _ = adaptive_tv(Yt, Mt, lam0, kap, a.n_outer, a.n_inner, a.grad_sigma,
                                    best_f[1], a.pilot_iters)
            s = psnr(Xt, xh.cpu()).mean()
            print(f"  [tune] ATV lam0={lam0:.3f} kappa={kap:.3f}  PSNR={s:.2f}")
            if s > best_a[0]:
                best_a = (s, lam0, kap)
    return best_f[1], best_a[1], best_a[2]


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) or a.device == "cuda" else "cpu")
    os.makedirs(a.out, exist_ok=True)
    src = f"image={a.image}" if a.image else f"dataset={a.dataset}"
    print(f"device={dev} {src} n={a.n_images} size={a.size} missing={a.missing} "
        f"mask={a.mask} sigma={a.sigma}")

    if a.image:
        X, names = load_user_images(a.image, a.n_images, a.size, gray=not a.color, fit=a.fit)
        # your own image(s) are never used for tuning: use bundled sample photos instead
        X_tune = load_images("skimage", a.n_tune, a.size, gray=not a.color) if a.tune else X[:0]
        print(f"Loaded {len(X)} image(s) -> {tuple(X.shape[1:])}: {', '.join(names)}")
    else:
        total = a.n_images + (a.n_tune if a.tune else 0)
        X_all = load_images(a.dataset, total, a.size, gray=not a.color, root=a.data_root)
        X, X_tune = X_all[:a.n_images], X_all[a.n_images:]
        names = [f"img{i:02d}" for i in range(len(X))]
    Y, M = degrade(X, a.missing, a.sigma, a.mask, a.block_size, a.seed)

    lam_ftv, lam0, kappa = a.lam_ftv, a.lam0, a.kappa
    if a.tune and len(X_tune) > 0:
        print("Tuning hyper-parameters on held-out images ...")
        Yt, Mt = degrade(X_tune, a.missing, a.sigma, a.mask, a.block_size, a.seed + 1)
        lam_ftv, lam0, kappa = tune(X_tune, Yt, Mt, a, dev)
        print(f"Selected: lam_ftv={lam_ftv}, lam0={lam0}, kappa={kappa}")

    Yd, Md = Y.to(dev), M.to(dev)
    recon, times = OrderedDict(), {}

    t0 = time.time()
    nf = nearest_fill(Yd, Md).clamp(0, 1).cpu()
    times["Nearest fill"] = time.time() - t0

    t0 = time.time()
    xf = fixed_tv(Yd, Md, lam_ftv, a.ftv_iters)
    times["Fixed TV"] = time.time() - t0

    t0 = time.time()
    xa, W, hist = adaptive_tv(Yd, Md, lam0, kappa, a.n_outer, a.n_inner, a.grad_sigma, lam_ftv, a.pilot_iters)
    times["Adaptive TV"] = time.time() - t0
    print("ATV objective per outer iteration (nonconvex, should decrease):",
        " -> ".join(f"{h:.3f}" for h in hist))

    recon["Fixed TV"], recon["Adaptive TV"] = xf.cpu(), xa.cpu()
    if a.hybrid:
        t0 = time.time()
        xd = dip_adaptive_tv(Yd, Md, xa, lam0, kappa, n_iter=a.dip_iters, lr=a.dip_lr,
                            grad_sigma=a.grad_sigma, device=dev, seed=a.seed)
        times["Adaptive TV + DIP"] = time.time() - t0
        recon["Adaptive TV + DIP"] = xd.cpu()

    results = OrderedDict([("Nearest fill", evaluate(X, nf))])
    for k, v in recon.items():
        results[k] = evaluate(X, v)
    print("\n" + summarize(results, times))

    # ---- figures
    plot_comparison_grid(X, Y, recon, results, os.path.join(a.out, "comparison_grid.png"),
                        n_show=min(len(X), 8))
    save_individual(os.path.join(a.out, "images"), names, X, Y, {"Nearest fill": nf, **recon})
    plot_weight_map(xa.cpu()[0], W.cpu()[0], os.path.join(a.out, "lambda_map.png"))
    # profile along the row of image 0 with the most horizontal-gradient energy (crosses many edges)
    row_energy = torch.abs(X[0, :, :, 1:] - X[0, :, :, :-1]).sum(dim=(0, 2))   # (H,)
    row = int(row_energy.argmax())
    plot_row_profile(X[0], {k: v[0] for k, v in recon.items()}, row, os.path.join(a.out, "row_profile.png"))

    # ---- machine-readable summary
    summary = {"args": vars(a),
                "selected": {"lam_ftv": lam_ftv, "lam0": lam0, "kappa": kappa},
                "atv_objective_history": hist,
                "metrics": {k: {m: {"mean": float(v.mean()), "std": float(v.std())} for m, v in r.items()}
                            for k, r in results.items()},
                "per_image": {nm: {k: {m: float(v[i]) for m, v in r.items()} for k, r in results.items()}
                                for i, nm in enumerate(names)},
                "time_s": times}
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if a.image:
        print("\nPer-image PSNR (dB):")
        for i, nm in enumerate(names):
            print(f"  {nm:<20}" + "  ".join(f"{k}: {r['psnr'][i]:.2f}" for k, r in results.items()))
    print(f"\nSaved to ./{a.out}/ : comparison_grid.png, lambda_map.png, row_profile.png, "
            f"metrics.json, images/<name>/*.png")


if __name__ == "__main__":
    main()
