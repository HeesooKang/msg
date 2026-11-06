import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from src.instruments import is_stock_symbol, load_instrument_groups, parse_master, tick_size, transaction_tax_rate
from src.strategies.momentum_scalp import MomentumScalpStrategy
from src.strategies.momentum_scalp_types import MomentumScalpConfig


class InstrumentTests(unittest.TestCase):
    def test_master_groups_drive_costs_and_ticks_not_watchlist_membership(self):
        content = "".join(
            symbol.ljust(9) + "KR7000000000" + "NAME" + group + " " * 225 + "\r\n"
            for symbol, group in (("152100", "EF"), ("18064K", "ST"), ("0000Z0", "ST"))
        ).encode("cp949")
        groups = parse_master(content, "kospi")
        self.assertEqual(groups, {"152100": "EF", "18064K": "ST", "0000Z0": "ST"})
        for invalid in ("Q500001", "ABCDEF", "00593", "18064k", "12345!", "１２３４５６"):
            self.assertFalse(is_stock_symbol(invalid), invalid)
        strategy = MomentumScalpStrategy(None, MomentumScalpConfig(inverse_etfs=[]))
        strategy.instrument_groups.update(groups)
        self.assertEqual(strategy._sell_tax_rate_for_symbol("152100"), 0)
        self.assertEqual(strategy._sell_tax_rate_for_symbol("18064K"), strategy.config.sell_tax_rate)
        self.assertEqual(tick_size(107000, groups["152100"]), 5)
        self.assertEqual(tick_size(107000, "ST"), 100)
        self.assertEqual(tick_size(1999, "EF"), 1)
        self.assertEqual(tick_size(2000, "EF"), 5)
        self.assertEqual(transaction_tax_rate("ST", .0015), .0015)

    def test_same_day_cache_needs_no_network(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "instruments.json"
            path.write_text(json.dumps({"date": "2026-09-16", "groups": {"152100": "EF"}}))
            with patch("src.instruments.requests.get") as request:
                self.assertEqual(load_instrument_groups(path, today=date(2026, 9, 16)), {"152100": "EF"})
                request.assert_not_called()
