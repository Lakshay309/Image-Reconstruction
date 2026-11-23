"""Regression tests for mask handling in the reconstruction solver."""
import torch
from solver import data_term, _split_observed_mask, adaptive_tv, _fit_dip


def test_data_term_ignores_targets_outside_mask():
    torch.manual_seed(4)
    x = torch.rand(1, 3, 12, 12)
    y1 = torch.rand_like(x)
    y2 = y1.clone()
    mask = (torch.rand(1, 1, 12, 12) > 0.5).float()
    y2 = torch.where(mask.bool(), y2, torch.rand_like(y2) * 100.0)
    assert torch.allclose(data_term(x, y1, mask), data_term(x, y2, mask))


def test_split_is_disjoint_and_covers_observations():
    torch.manual_seed(5)
    mask = (torch.rand(2, 1, 20, 20) > 0.3).float()
    train, val = _split_observed_mask(mask, 0.1, 99)
    assert torch.all(train * val == 0)
    assert torch.equal(train + val, mask)


def test_training_pilot_does_not_depend_on_heldout_values():
    torch.manual_seed(6)
    mask = (torch.rand(1, 1, 16, 16) > 0.25).float()
    train, val = _split_observed_mask(mask, 0.2, 123)
    y1 = mask * torch.rand(1, 1, 16, 16)
    y2 = y1.clone()
    y2 = torch.where(val.bool(), torch.full_like(y2, 50.0), y2)
    a1, _, _ = adaptive_tv(y1, train, 0.08, 0.05, n_outer=2, n_inner=8, pilot_iters=8)
    a2, _, _ = adaptive_tv(y2, train, 0.08, 0.05, n_outer=2, n_inner=8, pilot_iters=8)
    assert torch.allclose(a1, a2, atol=1e-6, rtol=0.0)


def test_dip_optimizer_has_nonzero_gradients_and_updates():
    torch.manual_seed(12)
    y = torch.rand(1, 1, 32, 32)
    mask = torch.ones(1, 1, 32, 32)
    train = mask.clone()
    val = torch.zeros_like(mask)
    val[:, :, ::4, ::4] = 1.0
    train = mask - val
    weights = torch.full_like(mask, 0.01)
    output, info = _fit_dip(
        y, train, weights, z_ch=4, base=8, depth=2,
        n_iter=8, lr=1e-3, noise_std=0.0, ema=0.0,
        eval_mask=val, eval_target=y, eval_every=2,
        patience=20, min_iters=2, min_delta=0.0,
        seed=12, device="cpu", reweight_every=0,
    )
    assert output is not None and output.shape == y.shape
    assert info["gradient_norm"] > 0.0, info
    assert info["parameter_max_update"] > 0.0, info
    assert len(info["validation_history"]) > 0


if __name__ == '__main__':
    test_data_term_ignores_targets_outside_mask()
    test_split_is_disjoint_and_covers_observations()
    test_training_pilot_does_not_depend_on_heldout_values()
    test_dip_optimizer_has_nonzero_gradients_and_updates()
    print('All solver regression tests passed.')
