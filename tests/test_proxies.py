"""Property-based tests for :mod:`src.scraper.proxies`.

Doctests on the pool cover concrete examples; these Hypothesis tests cover the
scheduling and eviction invariants that should hold for *any* pool size. A
constant ``clock`` keeps the per-proxy timing deterministic, and every pool is
built and exercised inside a single ``asyncio.run`` so its lock stays bound to
one event loop.
"""

import asyncio

from hypothesis import given
from hypothesis import strategies as st

from src.scraper.proxies import ProxyPool


def _urls(n: int) -> list[str]:
    """Return ``n`` distinct dummy proxy URLs."""
    return [f"http://10.0.0.{i}:8080" for i in range(n)]


@given(n=st.integers(min_value=1, max_value=25))
def test_reserves_each_proxy_once_before_repeating(n: int) -> None:
    """With all proxies idle, the first N reservations hit N distinct IPs."""

    async def run() -> list[str]:
        pool = ProxyPool(delay=2.0, min_size=0, clock=lambda: 0.0)
        pool.seed(_urls(n))
        seen = []
        for _ in range(n):
            reservation = await pool.reserve()
            assert reservation is not None
            url, wait = reservation
            assert wait == 0.0  # nothing is cooling down yet
            seen.append(url)
        return seen

    assert len(set(asyncio.run(run()))) == n


@given(n=st.integers(min_value=1, max_value=25))
def test_wait_after_exhaustion_equals_delay(n: int) -> None:
    """Once every proxy is reserved, the next one must wait a full ``delay``."""

    async def run() -> float:
        pool = ProxyPool(delay=2.0, min_size=0, clock=lambda: 0.0)
        pool.seed(_urls(n))
        for _ in range(n):
            await pool.reserve()
        reservation = await pool.reserve()
        assert reservation is not None
        return reservation[1]

    assert asyncio.run(run()) == 2.0


@given(failures=st.integers(min_value=0, max_value=8))
def test_evicts_only_after_max_consecutive_failures(failures: int) -> None:
    """A proxy survives until it accumulates ``max_failures`` in a row."""
    pool = ProxyPool(max_failures=3, min_size=0)
    pool.seed(["http://a:1"])
    for _ in range(failures):
        pool.report_bad("http://a:1")
    assert pool.size == (0 if failures >= 3 else 1)


@given(bad=st.integers(min_value=1, max_value=2))
def test_success_resets_failure_streak(bad: int) -> None:
    """A success clears prior failures, so eviction needs a fresh streak."""
    pool = ProxyPool(max_failures=3, min_size=0)
    pool.seed(["http://a:1"])
    for _ in range(bad):
        pool.report_bad("http://a:1")
    pool.report_ok("http://a:1")
    pool.report_bad("http://a:1")
    pool.report_bad("http://a:1")
    assert pool.size == 1  # only two failures since the reset
