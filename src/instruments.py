"""KIS security-group metadata, shared by execution prices and cost estimates."""

from datetime import date
from io import BytesIO
import json
import logging
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile

import requests


logger = logging.getLogger("kis_trader.instruments")
MASTER_ROOT = "https://new.real.download.dws.co.kr/common/master"


def is_stock_symbol(symbol: str) -> bool:
    # KRX short codes include letters (preferred shares and newer listings).
    # Q-prefixed ETN identifiers use a different instrument-code convention.
    return re.fullmatch(r"[0-9][0-9A-Z]{5}", symbol) is not None


def parse_master(content: bytes, market: str) -> dict[str, str]:
    # KIS stocks_info/{kis_kospi,kis_kosdaq}_code_mst.py: fixed tail includes LF.
    tail_width = {"kospi": 227, "kosdaq": 221}[market]
    groups = {}
    for line in content.decode("cp949").splitlines():
        if len(line) < 21 + tail_width:
            raise ValueError(f"invalid {market} master record")
        symbol = line[:9].strip()
        group = line[-tail_width:][:2]
        if is_stock_symbol(symbol):
            groups[symbol] = group
    if not groups:
        raise ValueError(f"empty {market} master")
    return groups


def load_instrument_groups(path: Path, *, today: date) -> dict[str, str]:
    """Refresh once per day, before streaming; never fetch per symbol or tick."""
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cached = {}
    groups = dict(cached.get("groups") or {})
    if cached.get("date") == today.isoformat() and groups:
        return groups
    try:
        refreshed = {}
        for market in ("kospi", "kosdaq"):
            response = requests.get(f"{MASTER_ROOT}/{market}_code.mst.zip", timeout=10)
            response.raise_for_status()
            with ZipFile(BytesIO(response.content)) as archive:
                refreshed.update(parse_master(archive.read(f"{market}_code.mst"), market))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"date": today.isoformat(), "groups": refreshed}), encoding="utf-8")
        temporary.replace(path)
        return refreshed
    except (requests.RequestException, OSError, ValueError, KeyError, BadZipFile) as exc:
        logger.warning("상품 분류 갱신 실패: 캐시 %d종목 사용 (%s)", len(groups), exc)
        return groups


def transaction_tax_rate(group: str, stock_rate: float) -> float:
    # Transaction tax only; fund income/withholding tax is not a trading fee.
    return 0.0 if group in {"EF", "EN"} else max(0.0, float(stock_rate))


def tick_size(price: float, group: str = "") -> int:
    if price < 2_000:
        return 1
    if group in {"EF", "EN"} or price < 5_000:
        return 5
    for ceiling, unit in ((20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500)):
        if price < ceiling:
            return unit
    return 1_000
