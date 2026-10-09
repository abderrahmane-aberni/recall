import time

import pytest

from app.rate_limit import RateLimitExceeded, RateLimiter


def test_allows_requests_under_the_cap():
    limiter = RateLimiter(per_ip_per_hour=3, global_per_day=100)
    for _ in range(3):
        limiter.check("1.2.3.4")  # should not raise


def test_blocks_once_per_ip_cap_is_hit():
    limiter = RateLimiter(per_ip_per_hour=2, global_per_day=100)
    limiter.check("1.2.3.4")
    limiter.check("1.2.3.4")
    with pytest.raises(RateLimitExceeded):
        limiter.check("1.2.3.4")


def test_per_ip_caps_are_independent():
    limiter = RateLimiter(per_ip_per_hour=1, global_per_day=100)
    limiter.check("1.1.1.1")
    with pytest.raises(RateLimitExceeded):
        limiter.check("1.1.1.1")
    limiter.check("2.2.2.2")  # different IP, should not raise


def test_global_cap_blocks_even_with_different_ips():
    limiter = RateLimiter(per_ip_per_hour=100, global_per_day=2)
    limiter.check("1.1.1.1")
    limiter.check("2.2.2.2")
    with pytest.raises(RateLimitExceeded):
        limiter.check("3.3.3.3")


def test_retry_after_is_positive_and_bounded():
    limiter = RateLimiter(per_ip_per_hour=1, global_per_day=100)
    limiter.check("1.2.3.4")
    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.check("1.2.3.4")
    assert 0 < exc_info.value.retry_after_seconds <= 3601


def test_old_hits_are_pruned_outside_the_window():
    limiter = RateLimiter(per_ip_per_hour=1, global_per_day=100)
    now = time.time()
    # simulate a hit from 2 hours ago by poking internal state directly
    limiter._per_ip_hits["1.2.3.4"].append(now - 7200)
    limiter.check("1.2.3.4")  # should not raise - the old hit is outside the 1h window
