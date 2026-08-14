import os
from datetime import date as date_cls, datetime


# Local safety net for dates where the KIS holiday endpoint is unavailable in
# paper trading. KIS remains the source of truth when reachable; these dates
# prevent weekday-only fallback from trading on known full-day KRX closures.
KRX_KNOWN_CLOSED_DATES = frozenset(
    {
        "20260101",
        "20260216",
        "20260217",
        "20260218",
        "20260302",
        "20260501",
        "20260505",
        "20260525",
        "20260603",
        "20260817",
        "20260924",
        "20260925",
        "20261005",
        "20261009",
        "20261225",
        "20261231",
    }
)

KRX_FIXED_ANNUAL_CLOSED_MMDD = frozenset(
    {
        "0101",
        "0501",
        "0505",
        "0606",
        "0815",
        "1003",
        "1009",
        "1225",
        "1231",
    }
)

EXTRA_CLOSED_DATE_ENV_KEYS = (
    "KIS_FORCE_CLOSED_DATES",
    "KRX_FORCE_CLOSED_DATES",
    "KIS_EXTRA_CLOSED_DATES",
    "KRX_EXTRA_CLOSED_DATES",
)


def normalize_trading_date(value=None) -> str:
    """Return YYYYMMDD for date-like values."""
    if value is None:
        return datetime.today().strftime("%Y%m%d")
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date_cls):
        return value.strftime("%Y%m%d")

    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) != 8:
        raise ValueError(f"invalid trading date: {value!r}")
    datetime.strptime(digits, "%Y%m%d")
    return digits


def _extra_closed_dates_from_env() -> set[str]:
    closed_dates: set[str] = set()
    for key in EXTRA_CLOSED_DATE_ENV_KEYS:
        raw_value = os.getenv(key, "")
        if not raw_value:
            continue
        for token in raw_value.replace(";", ",").replace(" ", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                closed_dates.add(normalize_trading_date(token))
            except ValueError:
                continue
    return closed_dates


def is_known_krx_closed_date(value=None) -> bool:
    date_key = normalize_trading_date(value)
    if date_key in _extra_closed_dates_from_env():
        return True
    if date_key in KRX_KNOWN_CLOSED_DATES:
        return True

    day = datetime.strptime(date_key, "%Y%m%d")
    return day.weekday() < 5 and date_key[4:] in KRX_FIXED_ANNUAL_CLOSED_MMDD


def is_krx_regular_trading_day(value=None) -> bool:
    date_key = normalize_trading_date(value)
    day = datetime.strptime(date_key, "%Y%m%d")
    return day.weekday() < 5 and not is_known_krx_closed_date(date_key)
