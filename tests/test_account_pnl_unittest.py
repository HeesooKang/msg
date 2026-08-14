import unittest

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
