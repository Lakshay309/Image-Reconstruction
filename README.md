# Adaptive TV + Deep Image Prior image reconstruction

This project compares **Nearest Fill**, **Fixed TV**, **Adaptive TV**, and **Adaptive-TV-regularized Deep Image Prior (ATV-DIP)** on user-supplied images.

No external image dataset is used to train the DIP generator. For each input image, a randomly initialized skip-connected convolutional generator is optimized from scratch against that image's observed pixels. No pretrained weights are loaded.

## Research basis and implementation choice

The most direct paper for this project is:

1. **Liu, Sun, Xu & Kamilov (2018), “Image Restoration using Total Variation Regularized Deep Image Prior.”** They parameterize the reconstruction as `x = f_theta(z)` and optimize the randomly initialized U-Net generator with a data-fidelity term plus an explicit TV penalty:

   `min_theta || y - H f_theta(z) ||_2^2 + lambda * TV(f_theta(z))`.

   Their reported denoising tables show DIP-TV outperforming both plain DIP and standalone TV on the tested Set12 cases/noise levels. This supports combining the network's implicit image prior with explicit TV; it does not guarantee superiority for every degradation or image.

2. **Ulyanov, Vedaldi & Lempitsky (2018), “Deep Image Prior.”** DIP uses a randomly initialized convolutional generator, fixed random input, and per-image optimization. Its architecture itself is the prior; it is not pretrained on a separate dataset. The method is sensitive to overfitting and stopping time.

3. **Liu et al. (ICCV 2023), “The Devil is in the Upsampling: Architectural Decisions Made Simpler for Denoising with Deep Image Prior.”** Their analysis emphasizes fixed/unlearnt upsampling and image-dependent depth/width. They report that architecture size alone is not enough; the upsampling structure strongly affects the spectral bias and noise fitting. This implementation uses nearest-neighbour unlearnt upsampling in a skip-connected encoder/decoder.

4. **Batard, Haro & Ballester, “DIP-VBTV: A Color Image Restoration Model Combining a Deep Image Prior and a Vector Bundle Total Variation,” SIAM Journal on Imaging Sciences.** This is a more geometric color prior (VBTV), not a drop-in replacement for ordinary adaptive TV. It reports improvements over DIP-VTV and unsupervised baselines for its studied restoration setting, but requires a more involved geometric regularizer.

5. **Liang et al. (2024), “Hybrid regularization inspired by total variation and deep denoiser prior for image restoration.”** This uses a plug-and-play pretrained FFDNet denoiser with TV and an adaptive balance parameter. It is not implemented here because it relies on an externally trained denoiser, whereas this project explicitly uses a per-image, untrained DIP and no external training data.

### Method implemented here

The hybrid is an **adaptive-TV-weighted DIP-TV objective**, not a pilot-output blend:

`x_theta = f_theta(z)`

`L(theta) = masked_MSE(x_theta, y; M_train) + lambda_DIP * sum_ij w_ij |D x_theta|_ij`

- `f_theta` is a randomly initialized DIP-style skip U-Net.
- `z` is fixed random input. Optional input perturbation is off by default because it can slow or destabilize optimization.
- `M_train` contains only observed training pixels; held-out observed pixels are excluded from the optimizer.
- `w_ij` is estimated from a training-only Adaptive TV pilot and periodically refreshed from the current DIP output. Large weights in flat regions and smaller weights near strong gradients preserve the adaptive-TV behavior.
- The old pilot anchor is intentionally removed. It is not part of the DIP-TV objective and can dominate or constrain the network's optimization.
- The held-out observed pixels select the iteration. The comparison is against a training-only Adaptive TV pilot, not a full-data result that has already seen validation pixels.
- If the DIP candidate does not beat the training-only pilot on held-out observations, the method honestly returns the full-data Adaptive TV baseline. Diagnostics state this explicitly. If it does beat it, the network is freshly refit using all observations for the selected iteration count.

This is a research-informed hybrid adaptation, not a claim that a paper has already published this exact combination or that it is universally best. The experiment must establish whether it helps for the user's degradation model.

## Installation

Use Python 3.10 or 3.11. Install PyTorch appropriate for your CUDA installation first if you need GPU acceleration, following the official PyTorch selector. Then:

```bash
pip install -r requirements.txt
```

## Run

```bash
python train_or_run.py --image "C:\\path\\to\\your\\image.png" --size 512 --color --fit crop --missing 0.10 --mask block --sigma 0.08 --block_size 8 --seed 0 --device cuda --out results --lam_ftv 0.08 --ftv_iters 400 --lam0 0.12 --kappa 0.05 --n_outer 6 --n_inner 150 --pilot_iters 300 --hybrid --dip_iters 1200 --dip_lr 0.001 --dip_patience 120 --dip_min_iters 100 --dip_eval_every 10 --dip_val_fraction 0.10 --dip_reweight_every 100 --dip_anchor_weight 0 --dip_noise_std 0 --dip_channels 16 --dip_base 16 --dip_depth 4 --ema 0.90
```

`--size` must be divisible by `2**dip_depth`. For 512 pixels and depth 4, this is satisfied.

## Outputs

The runner writes `comparison_grid.png`, `metrics.json`, `lambda_map.png`, `row_profile.png`, and per-method images under `results/images/`. `metrics.json` includes per-method PSNR, SSIM, edge-PSNR, runtimes, adaptive-TV objective history, and DIP training diagnostics (validation history, separate data/TV losses, gradient norm, parameter update magnitude, selection status, and whether full-data refitting occurred).

## Tests

```bash
python test_solver.py
python solver.py
```

Tests verify the mask fidelity term, disjoint train/validation split, no leakage into the training-only pilot, and nonzero DIP optimization updates on a small synthetic tensor. These are software tests, not evidence of image-quality superiority.

## References

- Ulyanov, D., Vedaldi, A., Lempitsky, V. (2018). *Deep Image Prior*. CVPR. https://arxiv.org/abs/1711.10925
- Liu, J., Sun, Y., Xu, X., Kamilov, U. S. (2018). *Image Restoration using Total Variation Regularized Deep Image Prior*. https://arxiv.org/abs/1810.12864
- Liu, Y. et al. (2023/2024). *The Devil is in the Upsampling: Architectural Decisions Made Simpler for Denoising with Deep Image Prior*. ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Liu_The_Devil_is_in_the_Upsampling_Architectural_Decisions_Made_Simpler_ICCV_2023_paper.html
- Batard, T., Haro, G., Ballester, C. *DIP-VBTV: A Color Image Restoration Model Combining a Deep Image Prior and a Vector Bundle Total Variation*. SIAM Journal on Imaging Sciences. https://doi.org/10.1137/20M1378697
- Liang, H. et al. (2024). *Hybrid regularization inspired by total variation and deep denoiser prior for image restoration*. https://doi.org/10.1007/s40747-024-01405-3
