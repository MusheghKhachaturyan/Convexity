"""
Python port of kpt sampler types (UniformSampler4, Bernoulli/Rectangular variants).
Uses NumPy; randomness matches the original intent (Fisher–Yates, binomial, piecewise linear).
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np


def _fisher_yates_partial(n: int, k: List[int], rng: random.Random) -> None:
    """UniformSampler::sample(N, k) — partial shuffle of first len(k) positions."""
    target = min(n, len(k))
    k[:] = list(range(target))
    samples_count = target
    if n <= 1:
        return
    displaced: dict[int, int] = {}
    for j in range(samples_count):
        idx = rng.randint(j, n - 1)
        if idx != j:
            if idx < samples_count:
                to_exchange = k[idx]
            else:
                to_exchange = displaced.setdefault(idx, idx)
            k[j], to_exchange = to_exchange, k[j]
            if idx < samples_count:
                k[idx] = to_exchange
            else:
                displaced[idx] = to_exchange


class Sampler(ABC):
    @abstractmethod
    def sample(self, k: List[int]) -> None:
        """Fill k with sampled indices (in-place)."""


class UniformSampler(Sampler):
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def sample(self, k: List[int]) -> None:
        raise NotImplementedError("Use sample_n(n, k)")

    def sample_n(self, n: int, k: List[int]) -> None:
        _fisher_yates_partial(n, k, self._rng)


class Sampler4(Sampler, ABC):
    samples_count = 4

    @abstractmethod
    def init(self, samples: Sequence[tuple[float, float]], win_size: tuple[int, int]) -> None:
        ...


class UniformSampler4(Sampler4):
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._n = 0

    def init(self, samples: Sequence[tuple[float, float]], win_size: tuple[int, int]) -> None:
        self._n = len(samples)
        _ = win_size

    def sample(self, k: List[int]) -> None:
        k[:] = list(range(self.samples_count))
        n = self._n
        if n <= 1:
            return
        displaced: dict[int, int] = {}
        for j in range(self.samples_count):
            idx = self._rng.randint(j, n - 1)
            if idx != j:
                if idx < self.samples_count:
                    to_exchange = k[idx]
                else:
                    to_exchange = displaced.setdefault(idx, idx)
                k[j], to_exchange = to_exchange, k[j]
                if idx < self.samples_count:
                    k[idx] = to_exchange
                else:
                    displaced[idx] = to_exchange


class BernoulliRandomSampler4(Sampler4):
    """Biased 4-point sampling using binomial offsets on corner-sorted index lists."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._m_tl: List[int] = []
        self._m_tr: List[int] = []

    def init(self, samples: Sequence[tuple[float, float]], win_size: tuple[int, int]) -> None:
        n = len(samples)
        w = win_size[0]
        distances_tl = np.empty(n, dtype=np.float64)
        distances_tr = np.empty(n, dtype=np.float64)
        for i, ptr in enumerate(samples):
            x, y = float(ptr[0]), float(ptr[1])
            yy = y * y
            distances_tl[i] = x * x + yy
            distances_tr[i] = (w - x) ** 2 + yy
        self._m_tl = list(np.argsort(distances_tl, kind="stable"))
        self._m_tr = list(np.argsort(distances_tr, kind="stable"))

    def _binomial(self, n_trials: int, p: float) -> int:
        return int(self._np_rng.binomial(n_trials, p))

    def sample(self, k: List[int]) -> None:
        k[:] = list(range(self.samples_count))
        n = len(self._m_tl)
        if n == 0:
            return
        order = n - 1
        displaced: dict[int, int] = {}
        for j in range(4):
            rng_val = self._binomial(order, 0.3) - order // 2
            folded = abs(rng_val)
            if j == 0:
                idx = self._m_tl[folded]
            elif j == 1:
                idx = self._m_tl[n - folded - 1]
            elif j == 2:
                idx = self._m_tr[folded]
            else:
                idx = self._m_tr[n - folded - 1]
            if idx != j:
                if idx < self.samples_count:
                    to_exchange = k[idx]
                else:
                    to_exchange = displaced.setdefault(idx, idx)
                k[j], to_exchange = to_exchange, k[j]
                if idx < self.samples_count:
                    k[idx] = to_exchange
                else:
                    displaced[idx] = to_exchange


class RectangularRandomSampler4(BernoulliRandomSampler4):
    """Piecewise-linear folded indices on tl/tr sorted lists."""

    def sample(self, k: List[int]) -> None:
        k[:] = list(range(self.samples_count))
        n = len(self._m_tl)
        if n == 0:
            return
        order = n // 2
        # C++ std::piecewise_linear_distribution on [0, order] with weights 0,1:
        # PDF ~ linear ramp; sampling via inverse CDF: x = order * sqrt(U).
        displaced: dict[int, int] = {}
        for j in range(4):
            u = self._rng.random()
            folded = int(order * math.sqrt(u)) if order > 0 else 0
            folded = min(folded, order)
            if j == 0:
                idx = self._m_tl[order - folded]
            elif j == 1:
                idx = self._m_tl[folded]
            elif j == 2:
                idx = self._m_tr[order - folded]
            else:
                idx = self._m_tr[folded]
            if idx != j:
                if idx < self.samples_count:
                    to_exchange = k[idx]
                else:
                    to_exchange = displaced.setdefault(idx, idx)
                k[j], to_exchange = to_exchange, k[j]
                if idx < self.samples_count:
                    k[idx] = to_exchange
                else:
                    displaced[idx] = to_exchange
