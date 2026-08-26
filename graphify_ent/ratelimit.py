"""Run as many concurrent model calls as the endpoint actually allows.

Measured against Ollama cloud on the 12-book ingest: six workers left the
endpoint idle (2.3 slices/min, ~7 h projected), twenty-four collected **83 HTTP
429s and lost 77 slices outright** — a fixed number cannot be right, because
the ceiling belongs to the service and moves with the hour.

So the ceiling is discovered instead of declared. Concurrency starts high,
narrows the moment the endpoint says 429, and widens again after a run of
clean calls. A rejected call is never lost: it waits for the window the server
names (`Retry-After`) or an exponential backoff with jitter, then goes again.

The jitter matters more than it looks. Without it, every worker rejected in the
same instant retries in the same instant, and the recovery is itself a burst
that earns another 429 — the thundering herd that turns one rate limit into a
sustained one.
"""

from __future__ import annotations

import random
import threading
import time

__all__ = ["AdaptiveLimiter"]


class AdaptiveLimiter:
    """A concurrency window that shrinks on rejection and grows on success.

    Not a token bucket: what is limited is how many calls are *in flight*,
    because that is what the endpoint pushes back on. `slot()` blocks until
    the window has room.
    """

    def __init__(self, start: int = 12, minimum: int = 2, maximum: int = 64,
                 widen_after: int = 12, shrink_factor: float = 0.6):
        if not 1 <= minimum <= maximum:
            raise ValueError("minimum must be between 1 and maximum")
        self.minimum = minimum
        self.maximum = maximum
        self.widen_after = widen_after
        self.shrink_factor = shrink_factor
        self._limit = max(minimum, min(start, maximum))
        self._in_flight = 0
        self._clean_streak = 0
        self._cv = threading.Condition()
        #: counters, for the run report — a limiter that silently throttles is
        #: indistinguishable from a slow endpoint.
        self.rejections = 0
        self.widenings = 0
        self.shrinks = 0

    @property
    def limit(self) -> int:
        with self._cv:
            return self._limit

    def _acquire(self) -> None:
        with self._cv:
            while self._in_flight >= self._limit:
                self._cv.wait(timeout=0.5)
            self._in_flight += 1

    def _release(self) -> None:
        with self._cv:
            self._in_flight -= 1
            self._cv.notify()

    def slot(self):
        """Context manager: occupies one place in the window."""
        limiter = self

        class _Slot:
            def __enter__(self):
                limiter._acquire()
                return limiter

            def __exit__(self, *exc):
                limiter._release()
                return False

        return _Slot()

    # -- feedback ---------------------------------------------------------
    def rejected(self, retry_after: float | None = None, attempt: int = 0) -> float:
        """Report a 429. Narrows the window and returns how long to wait.

        The server's own `Retry-After` wins when it sends one; otherwise an
        exponential backoff, always with jitter so a rejected cohort does not
        retry in lockstep.
        """
        with self._cv:
            self.rejections += 1
            self._clean_streak = 0
            new = max(self.minimum, int(self._limit * self.shrink_factor))
            if new < self._limit:
                self._limit = new
                self.shrinks += 1
            self._cv.notify_all()
        if retry_after and retry_after > 0:
            return retry_after + random.uniform(0, 1.5)
        return min(60.0, 2.0 * (2 ** attempt)) * random.uniform(0.6, 1.4)

    def succeeded(self) -> None:
        """Report a clean call. Widens the window after a run of them."""
        with self._cv:
            self._clean_streak += 1
            if self._clean_streak >= self.widen_after and self._limit < self.maximum:
                self._limit += 1
                self._clean_streak = 0
                self.widenings += 1
                self._cv.notify()

    def stats(self) -> dict:
        with self._cv:
            return {"limit_now": self._limit, "in_flight": self._in_flight,
                    "rejections": self.rejections, "shrinks": self.shrinks,
                    "widenings": self.widenings}


def sleep_with_jitter(seconds: float) -> None:
    time.sleep(max(0.0, seconds))
