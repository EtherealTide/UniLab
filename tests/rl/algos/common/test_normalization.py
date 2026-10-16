"""Tests for EmpiricalNormalization running-statistics updates.

Regression coverage for unilab_rl issue #77: the Welford cross-term in
``update_from_moments`` underestimated the population variance when batches
had different means, and made results depend on update order for unequal
batch sizes.
"""

from __future__ import annotations

import torch

from unilab.rl.algos.common.normalization import EmpiricalNormalization


class TestEmpiricalNormalizationUpdate:
    """Running-statistics correctness."""

    def test_shifted_constant_batches(self) -> None:
        """Issue #77 repro: two constant batches of 0s and 10s give var 25."""
        norm = EmpiricalNormalization(1, device="cpu")
        norm.update(torch.tensor([[0.0], [0.0]]))
        norm.update(torch.tensor([[10.0], [10.0]]))
        assert norm.count.item() == 4
        assert torch.allclose(norm.mean, torch.tensor([5.0]))
        assert torch.allclose(norm.std, torch.tensor([5.0]))

    def test_unequal_batch_sizes_order_independent(self) -> None:
        """Splitting one dataset into unequal batches is order-independent."""
        data = torch.randn(64, 3)

        def accumulate(first: slice, second: slice) -> EmpiricalNormalization:
            norm = EmpiricalNormalization(3, device="cpu")
            norm.update(data[first])
            norm.update(data[second])
            return norm

        forward = accumulate(slice(0, 17), slice(17, None))
        backward = accumulate(slice(17, None), slice(0, 17))
        for a, b in ((forward, backward),):
            assert torch.allclose(a.mean, b.mean, atol=1e-6)
            assert torch.allclose(a.std, b.std, atol=1e-6)
            assert a.count.item() == b.count.item()

        expected_mean = data.mean(dim=0)
        expected_var = data.var(dim=0, unbiased=False)
        assert torch.allclose(forward.mean, expected_mean, atol=1e-6)
        assert torch.allclose(forward.std, expected_var.sqrt(), atol=1e-5)

    def test_update_from_moments_matches_update(self) -> None:
        """Merging precomputed moments matches raw-sample updates."""
        data = torch.randn(50, 2)
        split = 11

        via_samples = EmpiricalNormalization(2, device="cpu")
        via_samples.update(data[:split])
        via_samples.update(data[split:])

        via_moments = EmpiricalNormalization(2, device="cpu")
        for part in (data[:split], data[split:]):
            via_moments.update_from_moments(
                part.mean(dim=0, keepdim=True),
                part.var(dim=0, keepdim=True, unbiased=False),
                part.shape[0],
            )

        assert via_moments.count.item() == via_samples.count.item()
        assert torch.allclose(via_moments.mean, via_samples.mean, atol=1e-6)
        assert torch.allclose(via_moments.std, via_samples.std, atol=1e-6)

    def test_checkpoint_resume_continues_statistics(self, tmp_path) -> None:
        """State-dict round-trip preserves buffers and later updates stay exact."""
        data = torch.randn(32, 4)
        split = 13

        uninterrupted = EmpiricalNormalization(4, device="cpu")
        uninterrupted.update(data[:split])
        uninterrupted.update(data[split:])

        resumed = EmpiricalNormalization(4, device="cpu")
        resumed.update(data[:split])
        checkpoint = tmp_path / "normalizer.pt"
        torch.save(resumed.state_dict(), checkpoint)

        loaded = EmpiricalNormalization(4, device="cpu")
        loaded.load_state_dict(torch.load(checkpoint, weights_only=True))
        loaded.update(data[split:])

        assert loaded.count.item() == uninterrupted.count.item()
        assert torch.allclose(loaded.mean, uninterrupted.mean, atol=1e-6)
        assert torch.allclose(loaded.std, uninterrupted.std, atol=1e-6)

    def test_frozen_inference_path_does_not_update(self) -> None:
        """Eval-mode forward normalizes without touching running statistics."""
        norm = EmpiricalNormalization(2, device="cpu")
        norm.update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        stats_before = norm.state_dict()

        norm.eval()
        x = torch.tensor([[10.0, -5.0]])
        out = norm(x)
        out_uncentered = norm(x, center=False)

        assert norm.state_dict().keys() == stats_before.keys()
        for key, value in stats_before.items():
            assert torch.equal(norm.state_dict()[key], value)
        assert torch.allclose(out, (x - norm._mean) / (norm._std + norm.eps), atol=1e-6)
        assert torch.allclose(out_uncentered, x / (norm._std + norm.eps), atol=1e-6)

    def test_training_forward_update_matches_manual_update(self) -> None:
        """The training-mode forward path uses the same update rule."""
        data = torch.tensor([[0.0], [0.0], [10.0], [10.0]])

        via_forward = EmpiricalNormalization(1, device="cpu")
        via_forward.train()
        via_forward(data[:2])
        via_forward(data[2:])

        via_update = EmpiricalNormalization(1, device="cpu")
        via_update.update(data[:2])
        via_update.update(data[2:])

        assert torch.allclose(via_forward.mean, via_update.mean, atol=1e-6)
        assert torch.allclose(via_forward.std, via_update.std, atol=1e-6)
