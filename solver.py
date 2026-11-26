
"""solver.py -- variational solvers for   min_x  1/2 ||M (.) x - y||^2 + R(x).

Fixed TV (FTV)       R(x) = lam * sum_ij ||grad x_ij||_2
Adaptive TV (ATV)    R(x) = sum_ij phi(||grad x_ij||_2),  phi(t) = lam0*kappa*log(1 + t/kappa)
                    solved by Majorize-Minimize = iteratively reweighted TV with weights
                    lam_ij^{k} = phi'(t_ij^{k}) = lam0 / (1 + ||grad x^{k}_ij|| / kappa).
Hybrid (ATV-DIP)     x = f_theta(z), with masked measurement fidelity plus adaptive weighted TV;
                     a DIP-style skip U-Net is optimized per image with Adam.

The standalone TV solvers use Chambolle-Pock PDHG. The DIP hybrid follows the DIP-TV
objective with spatially adaptive TV weights and held-out measurement selection; see README.md.
Tensors are (B, C, H, W); gradients are (B, C, 2, H, W); TV is *vectorial* (the 2-norm runs
over channels and both directions) so colour images do not bleed across channels.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from models import SmallUNet


# --------------------------------------------------------------------------------------
# Discrete operators: forward differences with Neumann boundary, and div = -grad^T
# --------------------------------------------------------------------------------------
def grad(x: torch.Tensor) -> torch.Tensor:
    gx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    gy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return torch.stack((gy, gx), dim=2)                                    # (B,C,2,H,W)


def div(p: torch.Tensor) -> torch.Tensor:
    """Negative adjoint of grad:  <grad x, p> = -<x, div p>."""
    qy, qx = p[:, :, 0, :-1, :], p[:, :, 1, :, :-1]
    dx = F.pad(qx, (0, 1, 0, 0)) - F.pad(qx, (1, 0, 0, 0))
    dy = F.pad(qy, (0, 0, 0, 1)) - F.pad(qy, (0, 0, 1, 0))
    return dx + dy


def grad_mag(x: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Pixelwise ||grad x_ij||_2 (over channels and directions) -> (B,1,H,W)."""
    return torch.sqrt((grad(x) ** 2).sum(dim=(1, 2)) + eps ** 2).unsqueeze(1)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    r = int(math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    C = x.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kx, groups=C)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), ky, groups=C)


# --------------------------------------------------------------------------------------
# Initialisation: fill missing pixels with the nearest observed value (scipy EDT)
# --------------------------------------------------------------------------------------
def nearest_fill(y: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    yn, Mn = y.detach().cpu().numpy(), M.detach().cpu().numpy()
    out = yn.copy()
    for b in range(yn.shape[0]):
        missing = Mn[b, 0] < 0.5
        if not missing.any() or missing.all():
            continue
        idx = ndi.distance_transform_edt(missing, return_distances=False, return_indices=True)
        for c in range(yn.shape[1]):
            out[b, c] = yn[b, c][idx[0], idx[1]]
    return torch.from_numpy(out).to(y)


# --------------------------------------------------------------------------------------
# Objectives (used for monitoring / proving monotone decrease)
# --------------------------------------------------------------------------------------
def data_term(x, y, M):
    # Mask the residual, not only x. Values of y outside M must have no effect.
    return 0.5 * (M * (x - y)).square().sum(dim=(1, 2, 3))


def ftv_objective(x, y, M, lam):
    return data_term(x, y, M) + lam * grad_mag(x).sum(dim=(1, 2, 3))


def adaptive_objective(x, y, M, lam0, kappa):
    t = grad_mag(x)
    return data_term(x, y, M) + (lam0 * kappa * torch.log1p(t / kappa)).sum(dim=(1, 2, 3))


# --------------------------------------------------------------------------------------
# Chambolle-Pock primal-dual for  min_x 1/2||Mx-y||^2 + sum_ij lam_ij ||(grad x)_ij||_2
# --------------------------------------------------------------------------------------
@torch.no_grad()
def pdhg_tv(y, M, lam, n_iter: int = 300, x0: torch.Tensor | None = None) -> torch.Tensor:
    """lam: float or tensor (B,1,H,W) of per-pixel weights."""
    tau = sigma = 0.99 / math.sqrt(8.0)               # tau*sigma*||grad||^2 < 1, ||grad||^2 <= 8
    lam_t = torch.as_tensor(lam, dtype=y.dtype, device=y.device)
    lam_t = lam_t.clamp_min(1e-12) if lam_t.dim() == 0 else lam_t.clamp_min(1e-12).unsqueeze(2)  # (B,1,1,H,W)

    x = nearest_fill(y, M) if x0 is None else x0.clone()
    x_bar = x.clone()
    p = torch.zeros_like(grad(x))
    for _ in range(n_iter):
        # dual ascent + projection onto the ball {||p_ij|| <= lam_ij}  (prox of conjugate of weighted TV)
        p = p + sigma * grad(x_bar)
        nrm = torch.sqrt((p ** 2).sum(dim=(1, 2), keepdim=True))
        p = p / torch.clamp(nrm / lam_t, min=1.0)
        # primal descent: x <- prox_{tau G}(x + tau div p), closed form for the masked quadratic
        x_old = x
        v = x + tau * div(p)
        x = (v + tau * M * y) / (1.0 + tau * M)
        x_bar = 2.0 * x - x_old                       # over-relaxation theta = 1
    return x


# --------------------------------------------------------------------------------------
# Baseline: fixed TV
# --------------------------------------------------------------------------------------
def fixed_tv(y, M, lam: float, n_iter: int = 400) -> torch.Tensor:
    return pdhg_tv(y, M, lam, n_iter).clamp(0, 1)


# --------------------------------------------------------------------------------------
# Spatially adaptive TV  (iteratively reweighted TV / MM)
# --------------------------------------------------------------------------------------
def adaptive_weights(x, lam0: float, kappa: float, grad_sigma: float = 0.0) -> torch.Tensor:
    """lam_ij = lam0 / (1 + ||grad x_ij|| / kappa)  -> (B,1,H,W).
    Large in flat regions (strong smoothing), small on edges (weak smoothing)."""
    return lam0 / (1.0 + grad_mag(gaussian_blur(x, grad_sigma)) / kappa)


@torch.no_grad()
def adaptive_tv(y, M, lam0: float, kappa: float, n_outer: int = 6, n_inner: int = 150,
                grad_sigma: float = 0.0, pilot_lam: float | None = None, pilot_iters: int = 300):
    """Returns (x_hat, final_weights, history of the nonconvex objective per outer iteration)."""
    x = pdhg_tv(y, M, lam0 if pilot_lam is None else pilot_lam, pilot_iters)   # pilot estimate
    hist = []
    for _ in range(n_outer):
        W = adaptive_weights(x, lam0, kappa, grad_sigma)                       # MM surrogate weights
        x = pdhg_tv(y, M, W, n_inner, x0=x)                                    # solve weighted-TV problem
        hist.append(adaptive_objective(x, y, M, lam0, kappa).mean().item())
    W = adaptive_weights(x, lam0, kappa, grad_sigma)
    return x.clamp(0, 1), W, hist


# --------------------------------------------------------------------------------------
# Adaptive-TV-regularized Deep Image Prior (ATV-DIP)
# --------------------------------------------------------------------------------------
def _split_observed_mask(M: torch.Tensor, val_fraction: float, seed: int):
    """Disjoint train/validation split of observed spatial pixels, per image."""
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_masks, val_masks = [], []
    for b in range(M.shape[0]):
        observed = (M[b, 0].detach().cpu() > 0.5).flatten()
        indices = torch.where(observed)[0]
        if indices.numel() < 2:
            raise ValueError("At least two observed pixels are required")
        order = torch.randperm(indices.numel(), generator=generator)
        n_val = min(max(1, int(round(indices.numel() * val_fraction))), indices.numel() - 1)
        val_idx, train_idx = indices[order[:n_val]], indices[order[n_val:]]
        vf = torch.zeros_like(observed, dtype=torch.float32)
        tf = torch.zeros_like(observed, dtype=torch.float32)
        vf[val_idx] = 1.0
        tf[train_idx] = 1.0
        val_masks.append(vf.reshape_as(M[b, 0]))
        train_masks.append(tf.reshape_as(M[b, 0]))
    val = torch.stack(val_masks).unsqueeze(1).to(device=M.device, dtype=M.dtype)
    train = torch.stack(train_masks).unsqueeze(1).to(device=M.device, dtype=M.dtype)
    return train, val


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(pred)
    return ((pred - target).square() * expanded).sum() / expanded.sum().clamp_min(1.0)


def _isotropic_tv_density(x: torch.Tensor) -> torch.Tensor:
    """Per-pixel vectorial/isotropic TV density, matching ``grad_mag``."""
    return grad_mag(x)


def _fit_dip(y, train_mask, W, *, z_ch, base, depth, n_iter, lr,
             noise_std, ema, eval_mask=None, eval_target=None, eval_every=10,
             patience=120, min_iters=100, min_delta=1e-5, seed=0,
             device="cpu", reweight_every=100, lam0=0.12, kappa=0.05,
             grad_sigma=0.0, tv_coefficient=0.12):
    """Optimize one randomly initialized DIP generator for one image."""
    dev = torch.device(device)
    torch.manual_seed(int(seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    _, channels, height, width = y.shape
    net = SmallUNet(z_ch, channels, base, depth=depth).to(dev)
    # Fixed random input, as in DIP. Optional small input perturbation is explicit.
    gen = torch.Generator(device=dev).manual_seed(int(seed) + 17)
    z = torch.rand((1, z_ch, height, width), generator=gen, device=dev) * 0.1
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_score = float("inf")
    best_out = None
    best_it = 0
    stale = 0
    history = []
    last_grad_norm = 0.0
    last_delta = 0.0
    last_components = {}
    avg = None
    for it in range(1, n_iter + 1):
        optimizer.zero_grad(set_to_none=True)
        zin = z + noise_std * torch.randn_like(z) if noise_std > 0 else z
        out = net(zin)
        data_loss = _masked_mse(out, y, train_mask)
        tv_map = _isotropic_tv_density(out)
        tv_loss = (W * tv_map).mean()
        loss = data_loss + tv_loss
        if not torch.isfinite(loss):
            break
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        last_grad_norm = float(grad_norm.detach().cpu())
        before = next(net.parameters()).detach().clone()
        optimizer.step()
        last_delta = float((next(net.parameters()).detach() - before).abs().max().cpu())
        with torch.no_grad():
            raw = net(z)
            avg = raw.detach() if avg is None else (ema * avg + (1.0 - ema) * raw.detach())
            if reweight_every > 0 and it % reweight_every == 0:
                W = (adaptive_weights(avg, lam0, kappa, grad_sigma) / max(lam0, 1e-12)).clamp(0.05, 1.0)
                W = W * tv_coefficient
            last_components = {
                "data_loss": float(data_loss.detach().cpu()),
                "tv_loss": float(tv_loss.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
                "gradient_norm": last_grad_norm,
                "parameter_max_update": last_delta,
                "output_mean": float(raw.mean().cpu()),
                "output_std": float(raw.std().cpu()),
            }
            if eval_mask is not None and (it % eval_every == 0 or it == n_iter):
                score = float(_masked_mse(avg, eval_target, eval_mask).cpu())
                history.append({"iteration": it, "validation_mse": score, **last_components})
                if score < best_score - min_delta:
                    best_score = score
                    best_out = avg.detach().clone()
                    best_it = it
                    stale = 0
                elif it >= min_iters:
                    stale += eval_every
                if it >= min_iters and stale >= patience:
                    break
    if eval_mask is None:
        best_out = avg.detach().clone() if avg is not None else None
        best_it = int(it if 'it' in locals() else 0)
    info = {
        "best_iteration": int(best_it),
        "best_validation_mse": float(best_score),
        "stopped_iteration": int(it if 'it' in locals() else 0),
        "validation_history": history,
        **last_components,
        "gradient_norm": last_grad_norm,
        "parameter_max_update": last_delta,
    }
    return best_out, info


def _dip_single(y, M, x_full_pilot, lam0, kappa, n_iter, lr, reweight_every,
                ema, grad_sigma, z_ch, base, depth, noise_std, dev, seed,
                val_fraction, patience, min_iters, min_delta, eval_every,
                tv_scale):
    y, M = y.to(dev), M.to(dev)
    x_full_pilot = x_full_pilot.to(dev).clamp(0, 1)
    train_mask, val_mask = _split_observed_mask(M, val_fraction, seed + 7919)

    # Validation baseline is computed ONLY from training observations. The
    # full-data ATV pilot is held back strictly as the final fallback.
    train_pilot, _, _ = adaptive_tv(
        y, train_mask, lam0, kappa, n_outer=4, n_inner=80,
        grad_sigma=grad_sigma, pilot_lam=lam0, pilot_iters=150)
    pilot_val = float(_masked_mse(train_pilot, y, val_mask).cpu())

    # Adaptive weights are estimated from the training-only pilot and normalized
    # to [0,1]. This is the adaptive-TV analogue of the explicit TV term in DIP-TV.
    weights = adaptive_weights(train_pilot, lam0, kappa, grad_sigma) / max(lam0, 1e-12)
    weights = weights.detach().clamp(0.05, 1.0)
    tv_coeff = float(lam0 if tv_scale is None else tv_scale)
    weights = weights * tv_coeff

    candidate, info = _fit_dip(
        y, train_mask, weights, z_ch=z_ch, base=base, depth=depth,
        n_iter=n_iter, lr=lr, noise_std=noise_std, ema=ema,
        eval_mask=val_mask, eval_target=y, eval_every=eval_every,
        patience=patience, min_iters=min_iters, min_delta=min_delta,
        seed=seed, device=dev, reweight_every=reweight_every,
        lam0=lam0, kappa=kappa, grad_sigma=grad_sigma, tv_coefficient=tv_coeff)

    info["training_pilot_validation_mse"] = pilot_val
    info["full_data_pilot_validation_mse"] = float(_masked_mse(x_full_pilot, y, val_mask).cpu())
    info["dip_beats_training_pilot"] = bool(
        candidate is not None and info["best_validation_mse"] < pilot_val - min_delta
    )
    info["dip_selected_on_validation"] = info["dip_beats_training_pilot"]
    info["full_data_refit"] = False
    if candidate is None or info["best_iteration"] <= 0:
        raise RuntimeError(
            "Adaptive TV + DIP optimization did not produce a valid validation-selected output. "
            "Inspect the loss/gradient diagnostics; no TV fallback was substituted."
        )

    # The pilot comparison is diagnostic only. Always refit the actual DIP
    # candidate using all observations for the validation-selected iteration count.
    full_weights = (adaptive_weights(x_full_pilot, lam0, kappa, grad_sigma) / max(lam0, 1e-12)).clamp(0.05, 1.0) * tv_coeff
    refit, refit_info = _fit_dip(
        y, M, full_weights, z_ch=z_ch, base=base, depth=depth,
        n_iter=max(1, info["best_iteration"]), lr=lr,
        noise_std=noise_std, ema=ema, eval_mask=None, eval_target=None,
        eval_every=eval_every, patience=patience, min_iters=min_iters,
        min_delta=min_delta, seed=seed, device=dev, reweight_every=reweight_every,
        lam0=lam0, kappa=kappa, grad_sigma=grad_sigma, tv_coefficient=tv_coeff)
    info["full_data_refit"] = True
    info["selected_method"] = "adaptive_tv_dip"
    info["refit_diagnostics"] = refit_info
    return refit.clamp(0, 1).detach().cpu(), info


def dip_adaptive_tv(
    y, M, x_pilot, lam0, kappa, n_iter: int = 1200, lr: float = 1e-3,
    reweight_every: int = 100, ema: float = 0.90, grad_sigma: float = 0.0,
    z_ch: int = 16, base: int = 16, depth: int = 4, noise_std: float = 0.0,
    device: str | torch.device | None = None, seed: int = 0,
    val_fraction: float = 0.10, patience: int = 120, min_iters: int = 100,
    min_delta: float = 1e-5, eval_every: int = 10,
    anchor_weight: float = 0.0, tv_scale: float | None = None,
    return_info: bool = False,
):
    """Adaptive-TV-regularized DIP with honest hold-out selection.

    The objective is masked MSE + weighted vectorial/isotropic TV. The generator is
    randomly initialized and optimized per image; no external dataset or
    pretrained network is used. A training-only ATV pilot is the validation
    comparator. The comparison is diagnostic only: a fresh DIP refit uses
    every observation regardless of whether DIP beats the pilot.
    ``anchor_weight`` remains as a deprecated compatibility argument and is
    intentionally not used: a pilot anchor would change the DIP-TV objective.
    """
    if not 0.0 <= ema < 1.0:
        raise ValueError("ema must be in [0,1)")
    if lr <= 0 or n_iter <= 0 or reweight_every <= 0:
        raise ValueError("lr, n_iter and reweight_every must be positive")
    if patience <= 0 or min_iters < 0 or eval_every <= 0:
        raise ValueError("patience/eval_every must be positive and min_iters non-negative")
    if noise_std < 0 or anchor_weight < 0:
        raise ValueError("noise_std and anchor_weight cannot be negative")
    if anchor_weight != 0:
        raise ValueError("DIP-TV implementation does not use an ATV pilot anchor; set --dip_anchor_weight 0")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    outputs, diagnostics = [], []
    for b in range(y.shape[0]):
        out, info = _dip_single(
            y[b:b+1], M[b:b+1], x_pilot[b:b+1], lam0, kappa,
            n_iter, lr, reweight_every, ema, grad_sigma, z_ch, base, depth,
            noise_std, dev, seed+b, val_fraction, patience, min_iters,
            min_delta, eval_every, tv_scale)
        outputs.append(out)
        diagnostics.append(info)
    result = torch.cat(outputs, dim=0)
    return (result, diagnostics) if return_info else result


# --------------------------------------------------------------------------------------
# DIP ablation baselines: DIP only and Fixed-TV + DIP
# --------------------------------------------------------------------------------------
def _dip_baseline_single(
    y, M, mode, lam, n_iter, lr, ema, z_ch, base, depth, noise_std,
    dev, seed, val_fraction, patience, min_iters, min_delta, eval_every,
):
    """DIP-only or fixed-TV-regularized DIP; baseline is diagnostic, never a fallback."""
    y, M = y.to(dev), M.to(dev)
    train_mask, val_mask = _split_observed_mask(M, val_fraction, seed + 7919)

    if mode == "dip_only":
        train_pilot = nearest_fill(y, train_mask).to(dev).clamp(0, 1)
        train_weights = torch.zeros_like(train_mask)
        full_weights = torch.zeros_like(M)
    elif mode == "fixed_tv_dip":
        train_pilot = fixed_tv(y, train_mask, lam, n_iter=300).to(dev)
        train_weights = torch.full_like(train_mask, float(lam))
        full_weights = torch.full_like(M, float(lam))
    else:
        raise ValueError(f"Unknown DIP baseline mode: {mode}")

    pilot_val = float(_masked_mse(train_pilot, y, val_mask).cpu())
    candidate, info = _fit_dip(
        y, train_mask, train_weights,
        z_ch=z_ch, base=base, depth=depth, n_iter=n_iter, lr=lr,
        noise_std=noise_std, ema=ema, eval_mask=val_mask, eval_target=y,
        eval_every=eval_every, patience=patience, min_iters=min_iters,
        min_delta=min_delta, seed=seed, device=dev, reweight_every=0,
        lam0=lam, kappa=0.05, grad_sigma=0.0, tv_coefficient=lam,
    )
    info["pilot_validation_mse"] = pilot_val
    info["dip_beats_baseline"] = bool(
        candidate is not None and info["best_validation_mse"] < pilot_val - min_delta
    )
    info["dip_selected_on_validation"] = info["dip_beats_baseline"]
    if candidate is None or info["best_iteration"] <= 0:
        raise RuntimeError(
            f"{mode} optimization did not produce a valid validation-selected output. "
            "Inspect the loss/gradient diagnostics; no baseline fallback was substituted."
        )

    # The baseline is diagnostic only; always refit and return the actual DIP
    # reconstruction so the ablation label corresponds to the method run.
    refit, refit_info = _fit_dip(
        y, M, full_weights,
        z_ch=z_ch, base=base, depth=depth,
        n_iter=max(1, info["best_iteration"]), lr=lr,
        noise_std=noise_std, ema=ema, eval_mask=None, eval_target=None,
        eval_every=eval_every, patience=patience, min_iters=min_iters,
        min_delta=min_delta, seed=seed, device=dev, reweight_every=0,
        lam0=lam, kappa=0.05, grad_sigma=0.0, tv_coefficient=lam,
    )
    info["full_data_refit"] = True
    info["selected_method"] = mode
    info["refit_diagnostics"] = refit_info
    return refit.clamp(0, 1).detach().cpu(), info


def dip_ablation(
    y, M, mode: str, lam: float = 0.08, n_iter: int = 1200,
    lr: float = 1e-3, ema: float = 0.90, z_ch: int = 16,
    base: int = 16, depth: int = 4, noise_std: float = 0.0,
    device=None, seed: int = 0, val_fraction: float = 0.10,
    patience: int = 120, min_iters: int = 150, min_delta: float = 1e-5,
    eval_every: int = 10, return_info: bool = False,
):
    """Run DIP-only or Fixed-TV+DIP, independently optimizing each image.

    The observed pixels are split for iteration selection; a fresh generator is
    then refit using all observations. Baselines are diagnostic only and never
    replace the DIP output. No external training dataset is used.
    """
    if mode not in {"dip_only", "fixed_tv_dip"}:
        raise ValueError("mode must be 'dip_only' or 'fixed_tv_dip'")
    if lam < 0 or n_iter <= 0 or lr <= 0:
        raise ValueError("lam must be non-negative; n_iter and lr must be positive")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    outputs, diagnostics = [], []
    for b in range(y.shape[0]):
        out, info = _dip_baseline_single(
            y[b:b+1], M[b:b+1], mode, lam, n_iter, lr, ema, z_ch,
            base, depth, noise_std, dev, seed + b, val_fraction,
            patience, min_iters, min_delta, eval_every,
        )
        outputs.append(out)
        diagnostics.append(info)
    result = torch.cat(outputs, dim=0)
    return (result, diagnostics) if return_info else result



# --------------------------------------------------------------------------------------
# Tiny self-test: python solver.py
# --------------------------------------------------------------------------------------
def _self_test():
    torch.manual_seed(0)
    x, p = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 2, 16, 16)
    lhs, rhs = (grad(x) * p).sum(), -(x * div(p)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-3), (lhs.item(), rhs.item())
    M = (torch.rand(1, 1, 16, 16) > 0.5).float()
    y = M * torch.rand(1, 1, 16, 16)
    xf = fixed_tv(y, M, 0.1, n_iter=10)
    xa, W, hist = adaptive_tv(y, M, 0.1, 0.05, n_outer=2, n_inner=5, pilot_iters=5)
    assert xf.shape == xa.shape == y.shape and W.shape == y.shape
    print("solver self-test passed")


if __name__ == "__main__":
    _self_test()
