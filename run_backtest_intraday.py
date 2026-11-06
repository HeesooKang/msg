#!/usr/bin/env python3
"""Replay the live execution policy from recorded quotes, not minute closes.

Usage: venv/bin/python run_backtest_intraday.py --date YYYY-MM-DD
       --root reports/market-quotes --output /private/tmp/policy-replay.json
No credentials, quote API or order API is used.
"""

from scripts.replay_execution_policy import main


if __name__ == "__main__":
    main()
