# An Adaptive Mathematical Framework for Image Recovery from Incomplete Information

## 1. Forward model

$$y = M \odot (x + \eta),\qquad \eta_{ij}\sim\mathcal N(0,\sigma^2)$$

$x\in[0,1]^{C\times H\times W}$ is the clean image and $M\in\{0,1\}^{H\times W}$ the sampling mask (0 = missing pixel).
On the observed set this equals $M\odot x+\eta$; on the missing set $y=0$ and carries no information, because the data term only involves $M\odot x$.
Recovering $x$ from $y$ is ill-posed (at 50% missing, the null space of $x\mapsto M\odot x$ is huge), so a prior is required.

## 2. Variational formulation

$$\hat x=\arg\min_x\ \tfrac12\|M\odot x-y\|_2^2+\mathcal R(x),\qquad \|\nabla x\|_{ij}=\Big(\sum_{c}\big[(\partial_h x_c)_{ij}^2+(\partial_v x_c)_{ij}^2\big]\Big)^{1/2}$$

(forward differences, Neumann boundary; vectorial TV so colour channels share edges).

**Fixed TV (baseline).** $\mathcal R_{\rm FTV}(x)=\lambda\sum_{ij}\|\nabla x\|_{ij}$ — one global weight.

**Spatially Adaptive TV (iteratively reweighted TV).** Use a *concave* potential of the gradient magnitude,

$$\mathcal R_{\rm ATV}(x)=\sum_{ij}\varphi(\|\nabla x\|_{ij}),\qquad \varphi(t)=\lambda_0\kappa\log\!\big(1+t/\kappa\big),\quad \varphi'(t)=\frac{\lambda_0}{1+t/\kappa}.$$

Because $\varphi$ is concave, it lies below its tangent at any $t_k$: $\varphi(t)\le\varphi(t_k)+\varphi'(t_k)(t-t_k)$. Majorize–Minimize therefore gives, at iterate $x^k$, a **weighted TV problem**

$$x^{k+1}=\arg\min_x\ \tfrac12\|M\odot x-y\|^2+\sum_{ij}\lambda^k_{ij}\|\nabla x\|_{ij},\qquad
\boxed{\lambda^k_{ij}=\frac{\lambda_0}{1+\|\nabla x^k\|_{ij}/\kappa}}$$

Flat regions ($\|\nabla x\|\approx0$) get $\lambda\approx\lambda_0$ (strong smoothing); edges ($\|\nabla x\|\gg\kappa$) get $\lambda\approx\lambda_0\kappa/\|\nabla x\|\ll\lambda_0$ (weak smoothing).
**Descent property:** $F(x^{k+1})\le S_k(x^{k+1})\le S_k(x^k)=F(x^k)$, where $S_k$ is the surrogate and $F$ the true (nonconvex) objective, so the objective decreases monotonically up to inner-solver accuracy. The code prints this history.
The pilot $x^0$ is the tuned FTV solution. Optional `--grad_sigma>0` smooths the gradient used for the weights (more noise-robust, but then the MM guarantee is only approximate).

## 3. Solver: Chambolle–Pock primal–dual (PDHG)

Each weighted-TV subproblem is the saddle problem $\min_x\max_p\ \langle\nabla x,p\rangle+G(x)-\iota_{\{\|p_{ij}\|\le\lambda_{ij}\}}(p)$ with $G(x)=\tfrac12\|M\odot x-y\|^2$. With $\tau=\sigma=0.99/\sqrt8$ (since $\|\nabla\|^2\le8$):

$$\begin{aligned}
p^{n+1}_{ij}&=\Pi_{\{\|\cdot\|\le\lambda_{ij}\}}\big(p^n_{ij}+\sigma(\nabla\bar x^n)_{ij}\big)=\frac{p^n_{ij}+\sigma(\nabla\bar x^n)_{ij}}{\max\{1,\ \|p^n_{ij}+\sigma(\nabla\bar x^n)_{ij}\|/\lambda_{ij}\}}\\
x^{n+1}&=\mathrm{prox}_{\tau G}\big(x^n+\tau\,\mathrm{div}\,p^{n+1}\big)=\frac{x^n+\tau\,\mathrm{div}\,p^{n+1}+\tau\,M\odot y}{1+\tau M}\quad(\text{elementwise, using }M^2=M)\\
\bar x^{n+1}&=2x^{n+1}-x^n
\end{aligned}$$

The masked-quadratic prox is closed-form, so no linear systems are solved; the iteration converges to a minimiser of the (convex) weighted-TV problem. `solver.py` verifies $\langle\nabla x,p\rangle=-\langle x,\mathrm{div}\,p\rangle$ numerically.

## 4. Hybrid with a neural prior (optional, `--hybrid`)

Replace the pixels by a Deep-Image-Prior generator $x=f_\theta(z)$ (small U-Net, fixed noise code $z$) and minimise

$$\min_\theta\ \tfrac12\|M\odot f_\theta(z)-y\|^2+\sum_{ij}\lambda_{ij}\sqrt{\|\nabla f_\theta(z)\|_{ij}^2+\epsilon^2}$$

with Adam; $\lambda_{ij}$ is refreshed every `reweight_every` iterations from an exponential moving average of the output (the same MM rule as above), and the pilot is the ATV result. The convolutional architecture supplies a learned-structure prior, the adaptive TV term controls noise fitting.

## 5. Why adaptivity helps (and what it does *not* do)

* **Bias / contrast loss.** In ROF-type models the TV term shrinks every gradient by an amount proportional to the local weight, independent of the gradient size. With a global $\lambda$, an edge of height $h$ loses contrast of order $\lambda$ and fine texture (small oscillations) is wiped out. With adaptive weights the loss at an edge is scaled by $\approx\kappa/\|\nabla x\|$, so edges and texture keep their contrast while flat noisy regions are still smoothed with $\lambda_0$. This is the same reason non-convex/log penalties reduce the bias of the $\ell_1$ norm.
* **Staircasing.** TV (an $\ell_1$ norm of the gradient) sets small gradients exactly to zero, so smooth ramps turn into piecewise-constant steps: this happens when $\lambda$ exceeds the "signal strength" of the local gradient. Reweighting lowers $\lambda$ where the pilot already shows a nonzero gradient (ramps, shading, textures) and keeps it high where the gradient is truly zero, so ramps are shrunk less and breaking into steps is reduced. Caveat: this relies on the pilot being reasonable, and it *reduces* rather than *eliminates* staircasing; second-order models (TGV) are the principled cure. Inspect `row_profile.png` to see the effect on your data.

## 6. Experimental protocol

* Metrics: PSNR, SSIM, and **edge-PSNR** (PSNR on the top-10% gradient pixels of the ground truth).
* Fairness: with `--tune`, FTV's $\lambda$ and ATV's $(\lambda_0,\kappa)$ are chosen by grid search on separate held-out images, so the baseline is as strong as possible.
* Use `--dataset skimage` or `stl10` for textures; CIFAR-10 upsampled from 32×32 is smooth, which understates edge/texture benefits.

## 7. Commands

```bash
pip install -r requirements.txt
python solver.py                                  # instant self-test (operators + shapes)
python train_or_run.py --dataset skimage --n_images 8 --tune        # no download needed
python train_or_run.py --dataset stl10 --n_images 8 --missing 0.6 --mask mixed --tune
python train_or_run.py --dataset cifar10 --n_images 8 --tune --hybrid --dip_iters 800
```

## 8. Running on your own image(s)  (`--image`)

```bash
# one image, 128x128, 50% random pixels missing
python train_or_run.py --image my_photo.jpg --size 128 --missing 0.5 --tune

# a whole folder (Set12, BSD68 crops, your photos ...), keep the full picture by padding
python train_or_run.py --image D:/pics/Set12 --n_images 12 --fit pad --size 128 --tune

# colour, block holes, plus the neural-prior hybrid
python train_or_run.py --image my_photo.png --color --mask block --missing 0.4 --hybrid
```

What happens, in order:

1. **Load & convert to the working size.** Every image (`.png .jpg .jpeg .bmp .tif .tiff .webp`; 8/16-bit, RGBA and grayscale are handled) is made square with `--fit crop` (centre crop, default, no distortion) or `--fit pad` (edge-pad, keeps the whole picture), resized to `--size` x `--size` (64 or 128 recommended), converted to grayscale (default) or RGB (`--color`), and scaled to [0,1]. This is your ground truth $x$.
2. **Degrade.** Random / block / mixed mask (`--mask`, `--missing`, `--block_size`) and Gaussian noise (`--sigma`): $y=M\odot(x+\eta)$.
3. **Reconstruct.** Nearest-fill reference, Fixed TV, Adaptive TV, and with `--hybrid` also Adaptive TV + DIP.
4. **Evaluate & save.** PSNR / SSIM / edge-PSNR against your original image.

With `--tune`, hyper-parameters are selected on bundled scikit-image sample photos, **never on your image**, so your result is not tuned to its own ground truth. (Without `--tune`, the defaults `--lam_ftv 0.08 --lam0 0.12 --kappa 0.05` are used.)
Use `--n_images N` to raise the cap on how many files of a folder are processed (default 8). Use `--seed` to change the random mask / noise.

### Outputs (`--out`, default `results/`)

| File | Content |
|---|---|
| `comparison_grid.png` | Original \| Corrupted ($y$) \| Fixed TV \| Adaptive TV (\| Adaptive TV + DIP), with PSNR/SSIM under each (up to 8 rows) |
| `images/<image_name>/original.png`, `corrupted.png`, `nearest_fill.png`, `fixed_tv.png`, `adaptive_tv.png`, (`adaptive_tv_dip.png`) | each result as its own full-size PNG, one folder per input image (corrupted: missing pixels are black) |
| `lambda_map.png` | the adaptive weight map $\lambda(i,j)$ for the first image |
| `row_profile.png` | intensity profile along an edge-rich row of the first image |
| `metrics.json` | settings, selected hyper-parameters, mean ± std metrics, **per-image metrics**, ATV objective history, timings |

Note: the saved PNGs are `--size` x `--size` pixels (small by design). Upscale them yourself for slides, e.g. with nearest-neighbour resampling, to avoid blurring the comparison.
