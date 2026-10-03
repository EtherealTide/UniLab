"""Curriculum learning for adaptive difficulty adjustment."""

from __future__ import annotations

import torch


class EpisodeLengthTracker:
    """Track a weighted moving average of episode lengths on one tensor."""

    def __init__(self, num_envs: int, window_size: int = 1000, *, device=None):
        self.num_envs = num_envs
        self.window_size = max(1, int(window_size * num_envs / 4096))
        device = torch.device("cpu") if device is None else torch.device(device)
        self._average = torch.zeros((), dtype=torch.float64, device=device)

    def update(self, episode_lengths: torch.Tensor) -> None:
        """Recursively update the average without synchronizing each batch."""
        count = int(episode_lengths.numel())
        if count == 0:
            return
        lengths = episode_lengths.to(dtype=torch.float64, device=self._average.device)
        batch_average = lengths.sum() / count
        weight = torch.as_tensor(
            min(count / self.window_size, 1.0),
            dtype=torch.float64,
            device=self._average.device,
        )
        self._average.add_((batch_average - self._average) * weight)

    @property
    def average(self) -> torch.Tensor:
        """Recursive average as a device-resident scalar tensor."""
        return self._average

    @property
    def average_length(self) -> float:
        """Public scalar view; only this boundary synchronizes."""
        return float(self._average.item())
