"""In-memory sliding-window rate limiting: per-IP and a global daily cap, to
protect the free-tier Gemini quota on a public demo. Same pattern as
CV-Maxxing's rate_limit.py.

In-memory means limits reset on every process restart/redeploy and don't
share state across multiple instances - fine for a single free-tier Render
service, not a substitute for a real rate-limit store if this ever scales
out.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict


class RateLimitExceeded(Exception):
    def __init__(self, message: str, retry_after_seconds: int) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after_seconds = retry_after_seconds


class RateLimiter:
    def __init__(self, per_ip_per_hour: int, global_per_day: int) -> None:
        self._per_ip_per_hour = per_ip_per_hour
        self._global_per_day = global_per_day
        self._per_ip_hits: dict[str, list[float]] = defaultdict(list)
        self._global_hits: list[float] = []
        self._lock = threading.Lock()

    def _prune(self, hits: list[float], window_seconds: float, now: float) -> list[float]:
        return [t for t in hits if now - t < window_seconds]

    def check(self, client_ip: str) -> None:
        now = time.time()
        with self._lock:
            self._global_hits = self._prune(self._global_hits, 86400, now)
            if len(self._global_hits) >= self._global_per_day:
                oldest = min(self._global_hits)
                retry_after = int(86400 - (now - oldest)) + 1
                raise RateLimitExceeded(
                    "This demo has hit its shared daily request limit. Please try again later.",
                    retry_after,
                )

            ip_hits = self._prune(self._per_ip_hits[client_ip], 3600, now)
            self._per_ip_hits[client_ip] = ip_hits
            if len(ip_hits) >= self._per_ip_per_hour:
                oldest = min(ip_hits)
                retry_after = int(3600 - (now - oldest)) + 1
                raise RateLimitExceeded(
                    "You've hit the per-hour request limit for this demo. Please try again later.",
                    retry_after,
                )

            self._global_hits.append(now)
            self._per_ip_hits[client_ip].append(now)
