import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.auth import TokenManager
from src.config import Config


def _config(
    api_key: str,
    api_secret: str,
    *,
    account_number: str = "12345678",
    account_product_code: str = "01",
) -> Config:
    return Config(
        trading_mode="paper",
        is_paper=True,
        api_key=api_key,
        api_secret=api_secret,
        account_number=account_number,
        account_product_code=account_product_code,
        base_url="https://openapivts.koreainvestment.com:29443",
        ws_url="ws://ops.koreainvestment.com:31000",
        rate_limit_interval=0.5,
        request_connect_timeout=3.05,
        request_read_timeout=10.0,
        log_level="INFO",
    )


class TokenManagerTests(unittest.TestCase):
    def test_token_cache_path_changes_when_paper_credentials_change(self):
        with TemporaryDirectory() as tmpdir, patch("src.auth.TOKEN_DIR", tmpdir):
            first = TokenManager(_config("paper-key-1", "paper-secret-1"))
            second = TokenManager(_config("paper-key-2", "paper-secret-2"))

            first_name = Path(first._token_file).name
            second_name = Path(second._token_file).name

        self.assertNotEqual(first_name, second_name)
        self.assertIn("KIS_paper_", first_name)
        self.assertNotIn("paper-key-1", first_name)
        self.assertNotIn("paper-secret-1", first_name)

    def test_token_file_stores_credential_fingerprint_not_raw_secret(self):
        with TemporaryDirectory() as tmpdir, patch("src.auth.TOKEN_DIR", tmpdir):
            manager = TokenManager(_config("paper-key", "paper-secret"))
            manager._save_token("token-value", "2099-01-01 00:00:00")
            payload = Path(manager._token_file).read_text(encoding="utf-8")

        self.assertIn("credential-fingerprint:", payload)
        self.assertNotIn("paper-key", payload)
        self.assertNotIn("paper-secret", payload)

    def test_token_cache_path_changes_when_account_changes(self):
        with TemporaryDirectory() as tmpdir, patch("src.auth.TOKEN_DIR", tmpdir):
            first = TokenManager(_config("paper-key", "paper-secret", account_number="12345678"))
            second = TokenManager(_config("paper-key", "paper-secret", account_number="87654321"))

            first_name = Path(first._token_file).name
            second_name = Path(second._token_file).name

        self.assertNotEqual(first_name, second_name)
        self.assertNotIn("12345678", first_name)
        self.assertNotIn("87654321", second_name)

    def test_token_file_without_matching_fingerprint_is_rejected(self):
        with TemporaryDirectory() as tmpdir, patch("src.auth.TOKEN_DIR", tmpdir):
            manager = TokenManager(_config("paper-key", "paper-secret"))
            Path(manager._token_file).write_text(
                "token: stale-token\nvalid-date: 2099-01-01 00:00:00\n",
                encoding="utf-8",
            )

            token = manager._load_token()

        self.assertIsNone(token)


if __name__ == "__main__":
    unittest.main()
