import unittest
from unittest.mock import Mock, patch

from src.account import AccountAPI


class DummyResponse:
    def __init__(self, success, output1=None, output2=None, error_message=""):
        self.success = success
        self.output1 = output1 or []
        self.output2 = output2 or []
        self.error_message = error_message
        self.has_next = False
        self.data = {}


class DummyClient:
    def __init__(self, response, *, is_paper=False):
        self.response = response
        self.last_get_kwargs = None
        self.config = type("Cfg", (), {"is_paper": is_paper})()

    def get(self, **kwargs):
        self.last_get_kwargs = kwargs
        return self.response


class AccountRealizedPnLTests(unittest.TestCase):
    def test_balance_timeout_reaches_every_page_without_changing_later_requests(self):
        first = DummyResponse(True, output1=[{"pdno": "005930", "hldg_qty": "2"}])
        first.has_next = True
        first.data = {"ctx_area_fk100": "next", "ctx_area_nk100": "page"}
        last = DummyResponse(
            True,
            output1=[{"pdno": "000660", "hldg_qty": "1"}],
            output2=[{"dnca_tot_amt": "1000000"}],
        )
        client = Mock(get=Mock(side_effect=[first, last, last]))
        account = AccountAPI(client)

        with patch("src.account.time.sleep"):
            balance = account.get_balance(timeout=(10.0, 60.0))
            account.get_balance()

        self.assertEqual(balance.total_deposit, 1_000_000)
        self.assertEqual([position.symbol for position in balance.positions], ["005930", "000660"])
        self.assertEqual(
            [call.kwargs["timeout"] for call in client.get.call_args_list],
            [(10.0, 60.0), (10.0, 60.0), None],
        )
        self.assertEqual(client.get.call_args_list[1].kwargs["tr_cont"], "N")

    def test_get_realized_profit_loss_reads_summary_output2(self):
        client = DummyClient(
            DummyResponse(
                success=True,
                output2=[{"rlzt_pfls": "-5,861"}],
            )
        )
        account = AccountAPI(client)

        pnl = account.get_realized_profit_loss()

        self.assertEqual(pnl, -5861)
        self.assertEqual(client.last_get_kwargs["tr_id"], "TTTC8494R")
        self.assertEqual(
            client.last_get_kwargs["api_url"],
            "/uapi/domestic-stock/v1/trading/inquire-balance-rlz-pl",
        )

    def test_get_realized_profit_loss_fallback_sums_output1(self):
        client = DummyClient(
            DummyResponse(
                success=True,
                output1=[
                    {"pdno": "005930", "rlzt_pfls": "1,200"},
                    {"pdno": "000660", "rlzt_pfls": "-300"},
                    {"pdno": "035420", "rlzt_pfls": "0"},
                ],
                output2=[{}],
            )
        )
        account = AccountAPI(client)

        pnl = account.get_realized_profit_loss()

        self.assertEqual(pnl, 900)

    def test_get_realized_profit_loss_returns_none_on_api_failure(self):
        client = DummyClient(
            DummyResponse(
                success=False,
                error_message="API error",
            )
        )
        account = AccountAPI(client)

        pnl = account.get_realized_profit_loss()

        self.assertIsNone(pnl)

    def test_get_realized_profit_loss_skips_api_in_paper_mode(self):
        client = DummyClient(
            DummyResponse(
                success=True,
                output2=[{"rlzt_pfls": "1,000"}],
            ),
            is_paper=True,
        )
        account = AccountAPI(client)

        pnl = account.get_realized_profit_loss()

        self.assertIsNone(pnl)
        self.assertIsNone(client.last_get_kwargs)


if __name__ == "__main__":
    unittest.main()
