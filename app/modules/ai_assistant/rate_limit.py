"""Giới hạn tần suất 15 tin nhắn/phút/user (§9.3). Bản in-memory cho 1 tiến trình; nhiều worker cần Redis (xem README)."""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Callable, Protocol


class RateLimiter(Protocol):
    def hit(self, key: str) -> tuple[bool, int]:
        """Trả (được_phép, retry_after_giây)."""
        ...


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit, self.window, self._clock = limit, window_s, clock
        self._hits: dict[str, deque[float]] = {}

    def hit(self, key: str) -> tuple[bool, int]:
        now = self._clock()
        q = self._hits.setdefault(key, deque())
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False, math.ceil(self.window - (now - q[0]))
        q.append(now)
        if len(self._hits) > 10_000:  # tránh phình bộ nhớ
            for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                self._hits.pop(k, None)
        return True, 0
