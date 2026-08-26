"""The concurrency window that discovers the endpoint's ceiling.

Measured on the 12-book ingest: 6 workers left the service idle, 24 collected
83 rejections and lost 77 slices. These tests pin the behaviour that replaces
the guess — narrow on 429, widen on a clean run, never lose the call.
"""

from __future__ import annotations

import threading

import pytest

from graphify_ent.ratelimit import AdaptiveLimiter


class TestWindow:
    def test_starts_within_bounds(self):
        assert AdaptiveLimiter(start=100, maximum=8).limit == 8
        assert AdaptiveLimiter(start=1, minimum=4, maximum=8).limit == 4

    def test_rejects_impossible_bounds(self):
        with pytest.raises(ValueError):
            AdaptiveLimiter(minimum=9, maximum=4)

    def test_admits_up_to_the_limit_and_blocks_beyond(self):
        lim = AdaptiveLimiter(start=2, minimum=1, maximum=2)
        held = [lim.slot().__enter__() for _ in range(2)]
        assert len(held) == 2
        entered = threading.Event()

        def third():
            with lim.slot():
                entered.set()

        t = threading.Thread(target=third, daemon=True)
        t.start()
        assert not entered.wait(timeout=0.3), "the third call must wait"
        lim._release()
        assert entered.wait(timeout=2), "and proceed once a slot frees"
        lim._release()


class TestFeedback:
    def test_rejection_narrows_the_window(self):
        lim = AdaptiveLimiter(start=20, minimum=2, maximum=32)
        lim.rejected()
        assert lim.limit == 12
        lim.rejected()
        assert lim.limit == 7

    def test_never_narrows_below_the_minimum(self):
        lim = AdaptiveLimiter(start=4, minimum=3, maximum=8)
        for _ in range(20):
            lim.rejected()
        assert lim.limit == 3

    def test_widens_only_after_a_clean_run(self):
        lim = AdaptiveLimiter(start=4, minimum=1, maximum=8, widen_after=3)
        lim.succeeded()
        lim.succeeded()
        assert lim.limit == 4, "two clean calls are not a run"
        lim.succeeded()
        assert lim.limit == 5

    def test_a_rejection_resets_the_clean_run(self):
        lim = AdaptiveLimiter(start=4, minimum=1, maximum=8, widen_after=3)
        lim.succeeded()
        lim.succeeded()
        lim.rejected()
        lim.succeeded()
        lim.succeeded()
        assert lim.limit <= 4, "the streak must restart after a rejection"

    def test_never_widens_past_the_maximum(self):
        lim = AdaptiveLimiter(start=2, minimum=1, maximum=3, widen_after=1)
        for _ in range(50):
            lim.succeeded()
        assert lim.limit == 3


class TestBackoff:
    def test_server_retry_after_is_honoured(self):
        lim = AdaptiveLimiter()
        wait = lim.rejected(retry_after=30.0)
        assert 30.0 <= wait <= 31.5, "the server's own window, plus jitter only"

    def test_backoff_grows_with_the_attempt(self):
        lim = AdaptiveLimiter()
        early = min(lim.rejected(attempt=0) for _ in range(20))
        late = max(lim.rejected(attempt=4) for _ in range(20))
        assert late > early

    def test_waits_are_jittered(self):
        """Without jitter a rejected cohort retries in lockstep and the
        recovery is itself the burst that earns the next 429."""
        lim = AdaptiveLimiter()
        waits = {round(lim.rejected(attempt=2), 4) for _ in range(15)}
        assert len(waits) > 1

    def test_backoff_is_capped(self):
        lim = AdaptiveLimiter()
        assert max(lim.rejected(attempt=20) for _ in range(10)) <= 60 * 1.4


class TestReporting:
    def test_counts_are_visible(self):
        """A limiter that silently throttles is indistinguishable from a slow
        endpoint; the run report has to be able to tell them apart."""
        lim = AdaptiveLimiter(start=4, minimum=1, maximum=8, widen_after=1)
        lim.rejected()
        lim.succeeded()
        s = lim.stats()
        assert s["rejections"] == 1 and s["shrinks"] == 1 and s["widenings"] == 1
        assert "limit_now" in s and "in_flight" in s
