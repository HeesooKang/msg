import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from src.scheduler import TradingScheduler


class _SequenceDateTime(datetime):
    timeline = []
    last = None

    @classmethod
    def now(cls, tz=None):
        if cls.timeline:
            cls.last = cls.timeline.pop(0)
            return cls.last
        if cls.last is None:
            raise AssertionError("datetime.now() 호출 시각이 준비되지 않았습니다.")
        return cls.last


class SchedulerTimingTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = TradingScheduler.__new__(TradingScheduler)
        self.scheduler.config = SimpleNamespace(off_hours_check_interval=1800)
        self.scheduler._shutdown = False

    def test_seconds_until_preopen_before_preopen_time(self):
        now = datetime(2026, 2, 27, 8, 49, 30)

        wait = self.scheduler._seconds_until_preopen(now)

        self.assertEqual(wait, 30)

    def test_seconds_until_preopen_during_preopen_waits_until_market_open(self):
        now = datetime(2026, 2, 27, 8, 50, 1)

        wait = self.scheduler._seconds_until_preopen(now)

        self.assertEqual(wait, 599)
        self.assertLess(wait, 3600)  # 다음날로 밀리지 않아야 한다.

    def test_is_trading_time_at_market_open_boundary(self):
        at_open = datetime(2026, 3, 5, 9, 0, 0)   # Thursday
        at_close = datetime(2026, 3, 5, 15, 30, 0)

        self.assertTrue(self.scheduler._is_trading_time(at_open))
        self.assertFalse(self.scheduler._is_trading_time(at_close))

    def test_is_trading_time_weekend_is_false(self):
        weekend_open = datetime(2026, 3, 7, 9, 0, 0)  # Saturday
        self.assertFalse(self.scheduler._is_trading_time(weekend_open))

    def test_sleep_until_preopen_returns_at_next_day_preopen_after_hardstop(self):
        sleep_calls = []

        def fake_sleep(seconds):
            sleep_calls.append(seconds)

        self.scheduler._interruptible_sleep = fake_sleep
        _SequenceDateTime.timeline = [
            datetime(2026, 3, 9, 15, 15, 13),
            datetime(2026, 3, 9, 15, 15, 13),
            datetime(2026, 3, 10, 8, 45, 0),
            datetime(2026, 3, 10, 8, 50, 1),
        ]
        _SequenceDateTime.last = None

        with patch("src.scheduler.datetime", _SequenceDateTime):
            self.scheduler._sleep_until_preopen()

        self.assertEqual(sleep_calls, [1800, 300])


if __name__ == "__main__":
    unittest.main()
