"""CPU reference for the experiment's page-retention policy.

This module deliberately has no torch/vLLM dependency. Runtime kernels must use
the same page-level ordering and protection rules. Visibility bounds are passed
explicitly: compact storage indexes are never logical token positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PolicyConfig:
    policy: Literal["off", "recent", "importance"] = "off"
    budget: int = 1024
    page_size: int = 16
    recent: int = 128
    native_window: int = 2048

    def __post_init__(self) -> None:
        if self.policy not in ("off", "recent", "importance"):
            raise ValueError(f"unknown drafter KV policy: {self.policy}")
        if self.page_size != 16:
            raise ValueError("the experiment uses 16-token pages")
        if self.budget <= 0 or self.budget % self.page_size:
            raise ValueError("budget must be a positive multiple of page_size")
        if self.recent <= 0 or self.recent > self.budget - self.page_size:
            raise ValueError("budget must fit recent tokens plus boundary overlap")
        if self.native_window <= 0:
            raise ValueError("native_window must be positive")

    @property
    def budget_pages(self) -> int:
        return self.budget // self.page_size

    @property
    def max_bootstrap_pages(self) -> int:
        # An unaligned W-token interval can intersect ceil((W + P - 1) / P) pages.
        return math.ceil((self.native_window + self.page_size - 1) / self.page_size)


def select_pages(
    scores: dict[int, float],
    *,
    config: PolicyConfig,
    newest_position: int,
    oldest_visible_position: int,
) -> tuple[int, ...]:
    """Return logical page IDs in chronological order, with no hidden backing.

    Scores are cumulative anchor-query probability mass, reduced across heads
    and layers. Only supplied pages can survive; an evicted page cannot be
    reconstructed. Ties prefer newer pages. Entire intersecting pages are
    protected, but attention must still mask individual out-of-window tokens.
    """
    if newest_position < oldest_visible_position:
        return ()
    if oldest_visible_position < 0:
        raise ValueError("oldest_visible_position must be nonnegative")
    for page, score in scores.items():
        if not isinstance(page, int) or page < 0:
            raise ValueError("logical page IDs must be nonnegative integers")
        if not math.isfinite(score) or score < 0:
            raise ValueError("page scores must be finite and nonnegative")
    size = config.page_size
    eligible = {
        page for page in scores
        if page * size <= newest_position
        and (page + 1) * size > oldest_visible_position
    }
    if config.policy == "off":
        return tuple(sorted(eligible))
    if config.policy == "recent":
        return tuple(sorted(sorted(eligible, reverse=True)[:config.budget_pages]))
    recent_start = max(oldest_visible_position, newest_position - config.recent + 1)
    protected = {page for page in eligible if (page + 1) * size > recent_start}
    if len(protected) > config.budget_pages:
        raise ValueError("protected pages exceed the persistent page budget")
    ranked = sorted(eligible - protected, key=lambda page: (scores[page], page), reverse=True)
    selected = protected | set(ranked[:config.budget_pages - len(protected)])
    return tuple(sorted(selected))


def retained_bytes(
    pages: int,
    *,
    layers: int = 5,
    kv_heads: int = 8,
    head_dim: int = 128,
    dtype_bytes: int = 2,
    page_size: int = 16,
) -> int:
    """Physical K+V payload; scores/tags and query/staging storage are separate."""
    if min(pages, layers, kv_heads, head_dim, dtype_bytes, page_size) < 0:
        raise ValueError("cache dimensions must be nonnegative")
    return 2 * pages * page_size * layers * kv_heads * head_dim * dtype_bytes
