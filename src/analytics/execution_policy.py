"""Observed top-of-book fills for offline replay, not a broker queue model."""

from src.models import Quote


def limit_fill(quote: Quote, limit: int, requested: int) -> tuple[int, int] | None:
    """None means depth was not observed, not a known failed order."""
    if not quote.has_valid_prices or not quote.book_available:
        return None
    if quote.ask_price <= 0 or quote.ask_price > limit:
        return (0, 0)
    if not quote.book_depth_available or quote.ask_size <= 0:
        return None
    return (min(max(0, requested), quote.ask_size), quote.ask_price)


def market_fill(quote: Quote, requested: int) -> tuple[int, int] | None:
    if (not quote.has_valid_prices or not quote.book_available
        or not quote.book_depth_available or not 0 < quote.bid_price <= quote.ask_price
        or quote.bid_size <= 0):
        return None
    return min(max(0, requested), quote.bid_size), quote.bid_price
