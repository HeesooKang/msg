"""모멘텀 스캘핑 전략.

매수: 모멘텀 점수(시가대비 상승, 등락률, 고가근접도, 거래량폭발) 기반
      + 시장 레짐 필터(KOSPI MA20)
매도: 익절(+1.5%) / 개별 손절(금액 기준 -5천원) / 추적손절(고점 -0.7%) / 장마감 청산
관리: 일일 목표 도달(순실현손익 ≥ +1만원) → 전량 청산 후 거래 중지
      일일 최대손실(순실현손익 ≤ -5천원) → 전량 청산 후 거래 중지
인버스: 약세 점수 ≥ 2일 때 인버스 ETF 매수 (공매도 효과)
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.market_data import MarketDataAPI
from src.models import Order, OrderResult, OrderSide, OrderType, Quote
from src.strategy import BaseStrategy

logger = logging.getLogger("kis_trader.strategy.momentum")

# 시가총액 상위 30종목 (하드코딩)
DEFAULT_STATIC_WATCHLIST = [
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "373220",  # LG에너지솔루션
    "207940",  # 삼성바이오로직스
    "005490",  # POSCO홀딩스
    "006400",  # 삼성SDI
    "051910",  # LG화학
    "035420",  # NAVER
    "000270",  # 기아
    "005380",  # 현대차
    "035720",  # 카카오
    "105560",  # KB금융
    "055550",  # 신한지주
    "012330",  # 현대모비스
    "066570",  # LG전자
    "003670",  # 포스코퓨처엠
    "028260",  # 삼성물산
    "032830",  # 삼성생명
    "003550",  # LG
    "086790",  # 하나금융지주
    "034730",  # SK
    "015760",  # 한국전력
    "017670",  # SK텔레콤
    "009150",  # 삼성전기
    "010130",  # 고려아연
    "033780",  # KT&G
    "018260",  # 삼성에스디에스
    "011200",  # HMM
    "138930",  # BNK금융지주
    "024110",  # 기업은행
]

# 인버스 ETF 유니버스
DEFAULT_INVERSE_ETFS = [
    "114800",  # KODEX 인버스 (KOSPI 200 역방향)
    "123310",  # TIGER 인버스 (KOSPI 200 역방향)
    "251340",  # KODEX 코스닥150 인버스
    "464930",  # TIGER 2차전지TOP10 인버스
]


@dataclass
class MomentumScalpConfig:
    """모멘텀 스캘핑 전략 설정."""

    seed_money: int = 1_000_000
    max_position_count: int = 5
    per_stock_amount: int = 200_000      # 종목당 기본 할당액
    max_per_stock_amount: int = 400_000  # 종목당 최대 노출 (피라미딩 상한)
    enable_pyramiding: bool = True
    scale_in_min_profit_pct: float = 0.3
    scale_in_score_bonus: float = 0.8

    # 일일 서킷 브레이커 (순실현손익 기준)
    daily_profit_target: int = 10_000    # 일일 목표 +1만원
    daily_loss_limit: int = -5_000       # 일일 최대손실 -5천원
    enable_unrealized_loss_guard: bool = True  # 미실현 포함 보조 손실컷 활성화
    daily_total_loss_limit: Optional[int] = None  # None이면 daily_loss_limit 사용

    # 개별 포지션 손절 (금액 기준)
    per_position_stop_loss: int = -5_000  # 포지션당 -5천원 즉시 청산

    # 익절 / 추적손절 (비율 기준)
    take_profit_pct: float = 1.5         # 익절 +1.5%
    trailing_stop_pct: float = -0.7      # 고점 대비 추적손절

    # 시장 레짐 필터 (KOSPI MA20 기반)
    bear_market_mode: str = 'A'          # 'A'=공격적(상대강도 종목만), 'B'=보수적(진입금지)

    # 거래 비용
    commission_rate: float = 0.00015     # 0.015% 수수료
    tax_slippage_rate: float = 0.002     # 0.20% 세금+슬리피지 (매도 시)

    # 종목 풀
    static_watchlist: List[str] = field(default_factory=lambda: DEFAULT_STATIC_WATCHLIST)
    dynamic_pool_size: int = 20
    pool_refresh_interval: int = 300     # 초

    # 필터
    min_change_rate: float = 0.5
    max_change_rate: float = 10.0
    min_volume: int = 100_000
    min_price: int = 1_000

    min_momentum_score: float = 2.0

    # 재매수 쿨다운
    cooldown_seconds: int = 600          # 10분

    # === 인버스 ETF 설정 ===
    inverse_enabled: bool = True
    inverse_etfs: List[str] = field(default_factory=lambda: DEFAULT_INVERSE_ETFS)
    inverse_max_positions: int = 2           # 인버스 최대 보유 수
    inverse_take_profit_pct: float = 1.0     # 인버스 익절 +1.0% (일반보다 빠르게)
    inverse_stop_loss_pct: float = -0.5      # 인버스 손절 -0.5% (타이트)
    inverse_trailing_stop_pct: float = -0.3  # 인버스 추적손절 (고점 -0.3%)
    inverse_max_hold_minutes: int = 120      # 최대 2시간 보유 (음의 복리 방지)
    bearish_threshold: int = 2               # 이 점수 이상일 때 인버스 진입
    inverse_min_momentum: float = 1.5        # 인버스 매수 최소 모멘텀 점수


@dataclass
class PositionState:
    """보유 포지션 상태."""
    symbol: str
    buy_price: int
    quantity: int
    invested_amount: int = 0
    buy_time: datetime = field(default_factory=datetime.now)
    high_since_buy: int = 0

    def __post_init__(self):
        if self.invested_amount <= 0:
            self.invested_amount = self.buy_price * self.quantity
        if self.high_since_buy == 0:
            self.high_since_buy = self.buy_price


@dataclass
class DailyPnL:
    """일일 손익 추적."""
    realized_gross_pnl: int = 0
    realized_net_pnl: int = 0
    fees_paid: int = 0
    taxes_paid: int = 0
    trade_count: int = 0

    @property
    def realized_pnl(self) -> int:
        """하위 호환용 alias: 순실현손익."""
        return self.realized_net_pnl

    @property
    def total_pnl(self) -> int:
        return self.realized_net_pnl


class MomentumScalpStrategy(BaseStrategy):
    """모멘텀 스캘핑 전략."""

    def __init__(
        self,
        market_data: MarketDataAPI,
        config: MomentumScalpConfig = None,
        pool_override: List[str] = None,
    ):
        self.market_data = market_data
        self.cfg = config or MomentumScalpConfig()
        self._pool_override = pool_override

        self.positions: Dict[str, PositionState] = {}
        self.daily_pnl = DailyPnL()
        self._pool: List[str] = []
        self._last_pool_refresh: Optional[datetime] = None
        self._halted = False
        self._avg_volumes: Dict[str, int] = {}
        self._quotes_cache: Dict[str, Quote] = {}
        self._sell_cooldown: Dict[str, datetime] = {}
        self._bear_score: int = 0
        self._bear_market = False
        self._inverse_symbols: set = set(self.cfg.inverse_etfs)
        self._halt_date: Optional[date] = None
        self._current_day: Optional[date] = None

    def initialize(self):
        today = datetime.now().date()
        if self._current_day != today:
            self.daily_pnl = DailyPnL()
            self._halted = False
            self._halt_date = None
            self._sell_cooldown = {}
            self._current_day = today
        self._build_pool()
        self._check_market_regime()

        if self._halted and self._halt_date == today:
            logger.info("당일 하드스탑 상태 유지: 신규 거래 중지")

        logger.info("전략 초기화: 모멘텀 스캘핑")
        logger.info("  시드: %s원, 종목당: %s원",
                     f"{self.cfg.seed_money:,}", f"{self.cfg.per_stock_amount:,}")
        logger.info("  익절: +%.1f%%, 개별손절: %s원, 추적손절: %.1f%%",
                     self.cfg.take_profit_pct,
                     f"{self.cfg.per_position_stop_loss:,}",
                     self.cfg.trailing_stop_pct)
        logger.info("  일일 목표(순실현): +%s원, 최대손실(순실현): %s원",
                     f"{self.cfg.daily_profit_target:,}",
                     f"{self.cfg.daily_loss_limit:,}")
        if self.cfg.enable_unrealized_loss_guard:
            total_loss_limit = (
                self.cfg.daily_total_loss_limit
                if self.cfg.daily_total_loss_limit is not None
                else self.cfg.daily_loss_limit
            )
            logger.info("  보조손실컷(순손익추정): %s원", f"{total_loss_limit:,}")
        logger.info("  시장 레짐: 약세점수=%d (모드: %s)",
                     self._bear_score, self.cfg.bear_market_mode)
        if self.cfg.inverse_enabled:
            logger.info("  인버스: 활성화 (임계=%d, 최대%d종목)",
                         self.cfg.bearish_threshold, self.cfg.inverse_max_positions)
        logger.info("  풀 크기: %d종목", len(self._pool))

    def get_watchlist(self) -> List[str]:
        now = datetime.now()
        if (self._last_pool_refresh and
                (now - self._last_pool_refresh).total_seconds() >= self.cfg.pool_refresh_interval):
            self._build_pool()
        return self._pool

    def on_tick(self, quote: Quote) -> List[Order]:
        self._quotes_cache[quote.symbol] = quote
        orders = []

        if self._halted:
            return orders

        if quote.symbol in self.positions:
            order = self._evaluate_sell(quote)
            if order:
                orders.append(order)
            else:
                scale_in = self._evaluate_buy(quote)
                if scale_in:
                    orders.append(scale_in)

        else:
            long_count = sum(1 for s in self.positions if s not in self._inverse_symbols)
            if long_count < self.cfg.max_position_count:
                order = self._evaluate_buy(quote)
                if order:
                    orders.append(order)

        return orders

    def on_batch_tick(self, quotes: List[Quote]) -> List[Order]:
        """배치 시세를 받아 전체적으로 판단한다."""
        for q in quotes:
            self._quotes_cache[q.symbol] = q

        # 거래 중지 상태면 주문 없음
        if self._halted:
            return []

        # 미실현 손익 업데이트 + 고가 추적
        for sym, pos in self.positions.items():
            q = self._quotes_cache.get(sym)
            if q:
                if q.current_price > pos.high_since_buy:
                    pos.high_since_buy = q.current_price

        # 장마감 청산 (15:15 이후) — 반드시 halt 설정 (실거래 모드만)
        if self.market_data is not None:
            now = datetime.now()
            if now.hour >= 15 and now.minute >= 15:
                self._halted = True
                self._halt_date = now.date()
                if self.positions:
                    logger.info("장마감 임박 → 전량 청산")
                    return self._liquidate_all()
                return []

        # 서킷 브레이커: 순실현손익 기준으로 판단
        realized_net = self.daily_pnl.realized_net_pnl

        if realized_net <= self.cfg.daily_loss_limit:
            logger.warning(
                "일일 손실한도 도달! (순실현: %s원) → 전량 청산 후 거래 중지",
                f"{realized_net:,}",
            )
            self._halted = True
            self._halt_date = datetime.now().date()
            return self._liquidate_all()

        if realized_net >= self.cfg.daily_profit_target:
            logger.info(
                "일일 목표 달성! (순실현: %s원) → 전량 청산 후 거래 중지",
                f"{realized_net:,}",
            )
            self._halted = True
            self._halt_date = datetime.now().date()
            return self._liquidate_all()

        # 보조 손실컷: 순손익 추정(순실현 + 미실현 추정) 기준
        if self.cfg.enable_unrealized_loss_guard:
            total_loss_limit = (
                self.cfg.daily_total_loss_limit
                if self.cfg.daily_total_loss_limit is not None
                else self.cfg.daily_loss_limit
            )
            unrealized_net = self._estimate_unrealized_net_pnl()
            total_net = realized_net + unrealized_net
            if total_net <= total_loss_limit:
                logger.warning(
                    "보조 손실컷 도달! (순실현: %s원, 미실현추정: %s원, 합계: %s원) "
                    "→ 전량 청산 후 거래 중지",
                    f"{realized_net:,}",
                    f"{unrealized_net:,}",
                    f"{total_net:,}",
                )
                self._halted = True
                self._halt_date = datetime.now().date()
                return self._liquidate_all()

        # 백테스트 모드: 배치 시세에서 약세 점수 추정
        if self.market_data is None:
            self._estimate_market_from_quotes(quotes)

        # 개별 종목 평가
        orders = []

        # 1) 매도 먼저 (일반 + 인버스 모두)
        for q in quotes:
            if q.symbol in self.positions:
                if q.symbol in self._inverse_symbols:
                    order = self._evaluate_inverse_sell(q)
                else:
                    order = self._evaluate_sell(q)
                if order:
                    orders.append(order)

        # 2) 일반 매수 (롱 포지션 카운트 기준)
        for q in quotes:
            if q.symbol in self._inverse_symbols:
                continue  # 인버스는 아래에서 별도 처리
            long_count = sum(
                1 for s in self.positions if s not in self._inverse_symbols
            )
            pending_long = sum(
                1 for o in orders
                if (
                    o.side == OrderSide.BUY and
                    o.symbol not in self._inverse_symbols and
                    o.symbol not in self.positions
                )
            )
            if q.symbol not in self.positions and long_count + pending_long >= self.cfg.max_position_count:
                continue

            order = self._evaluate_buy(q, pending_orders=orders)
            if order:
                orders.append(order)

        # 3) 인버스 매수 (인버스 포지션 카운트 기준)
        if self.cfg.inverse_enabled and self._bear_score >= self.cfg.bearish_threshold:
            for q in quotes:
                if q.symbol not in self._inverse_symbols:
                    continue
                inv_count = sum(
                    1 for s in self.positions if s in self._inverse_symbols
                )
                pending_inv = sum(
                    1 for o in orders
                    if o.side == OrderSide.BUY and o.symbol in self._inverse_symbols
                )
                if inv_count + pending_inv >= self.cfg.inverse_max_positions:
                    break
                if q.symbol not in self.positions:
                    order = self._evaluate_inverse_buy(q, pending_orders=orders)
                    if order:
                        orders.append(order)

        return orders

    def on_order_filled(self, result: OrderResult):
        if result.side == OrderSide.BUY:
            if not result.success:
                return

            fill_price = result.price
            if fill_price <= 0:
                cached = self._quotes_cache.get(result.symbol)
                fill_price = cached.current_price if cached else 0
            if fill_price <= 0:
                return

            buy_notional = fill_price * result.quantity
            buy_fee = self._calc_commission_cost(buy_notional)
            if buy_fee > 0:
                self.daily_pnl.fees_paid += buy_fee
                # 매수 수수료는 체결 시점에 확정 비용으로 반영
                self.daily_pnl.realized_net_pnl -= buy_fee

            existing = self.positions.get(result.symbol)
            if existing:
                total_qty = existing.quantity + result.quantity
                total_invested = existing.invested_amount + (fill_price * result.quantity)
                existing.quantity = total_qty
                existing.invested_amount = total_invested
                existing.buy_price = int(round(total_invested / total_qty))
                if fill_price > existing.high_since_buy:
                    existing.high_since_buy = fill_price
                tag = "[INV] " if result.symbol in self._inverse_symbols else ""
                logger.info(
                    "%s추가매수 체결: %s +%d주 @ %s원 (평단 %s원, 총 %d주)",
                    tag,
                    result.symbol,
                    result.quantity,
                    f"{fill_price:,}",
                    f"{existing.buy_price:,}",
                    existing.quantity,
                )
                return

            self.positions[result.symbol] = PositionState(
                symbol=result.symbol,
                buy_price=fill_price,
                quantity=result.quantity,
                invested_amount=fill_price * result.quantity,
            )
            tag = "[INV] " if result.symbol in self._inverse_symbols else ""
            logger.info("%s매수 체결: %s %d주 @ %s원",
                        tag, result.symbol, result.quantity, f"{fill_price:,}")

        elif result.side == OrderSide.SELL:
            if not result.success:
                # 매도 실패 시 실제 보유는 유지되므로 포지션을 제거하면 안 된다.
                logger.warning("매도 실패(포지션 유지): %s", result.symbol)
                return

            pos = self.positions.pop(result.symbol, None)
            if pos:
                sell_price = result.price
                if sell_price <= 0:
                    cached = self._quotes_cache.get(result.symbol)
                    sell_price = cached.current_price if cached else pos.buy_price

                gross_pnl = (sell_price - pos.buy_price) * pos.quantity
                sell_notional = sell_price * pos.quantity
                sell_fee = self._calc_commission_cost(sell_notional)
                sell_tax_slippage = self._calc_sell_tax_slippage_cost(sell_notional)
                net_pnl = gross_pnl - sell_fee - sell_tax_slippage

                self.daily_pnl.realized_gross_pnl += gross_pnl
                self.daily_pnl.realized_net_pnl += net_pnl
                self.daily_pnl.fees_paid += sell_fee
                self.daily_pnl.taxes_paid += sell_tax_slippage
                self.daily_pnl.trade_count += 1

                self._sell_cooldown[result.symbol] = datetime.now()

                tag = "[INV] " if result.symbol in self._inverse_symbols else ""
                logger.info(
                    "%s매도 체결: %s %d주 @ %s원 "
                    "(총손익: %s원, 순손익: %s원, 누적순손익: %s원)",
                    tag, result.symbol, result.quantity, f"{sell_price:,}",
                    f"{gross_pnl:,}", f"{net_pnl:,}", f"{self.daily_pnl.realized_net_pnl:,}",
                )

    def should_continue(self) -> bool:
        if self._halted and not self.positions:
            return False
        return True

    # --- 내부 로직 ---

    def _build_pool(self):
        """종목 풀을 구성한다."""
        pool = set(self.cfg.static_watchlist)

        if self._pool_override:
            self._pool = list(self._pool_override)
            # 인버스 ETF가 override에 없으면 추가
            if self.cfg.inverse_enabled:
                for sym in self.cfg.inverse_etfs:
                    if sym not in self._pool:
                        self._pool.append(sym)
            self._last_pool_refresh = datetime.now()
            return

        # 인버스 ETF 추가
        if self.cfg.inverse_enabled:
            for sym in self.cfg.inverse_etfs:
                pool.add(sym)

        if self.market_data:
            try:
                rising = self.market_data.get_fluctuation_ranking(
                    count=self.cfg.dynamic_pool_size,
                    min_change_rate=self.cfg.min_change_rate,
                    max_change_rate=self.cfg.max_change_rate,
                    min_price=self.cfg.min_price,
                    min_volume=self.cfg.min_volume,
                )
                for item in rising:
                    pool.add(item.symbol)
                logger.info("동적 풀 갱신: 등락률 상위 %d개 추가 (총 %d종목)",
                            len(rising), len(pool))
            except Exception as e:
                logger.warning("등락률 순위 조회 실패, 정적 풀만 사용: %s", e)

        self._pool = list(pool)[:55]  # 인버스 포함하여 여유 확보
        self._last_pool_refresh = datetime.now()

    def _check_market_regime(self):
        """KOSPI 기반 약세 점수(0~3)를 계산한다."""
        if not self.market_data:
            self._bear_score = 0
            self._bear_market = False
            return

        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=45)).strftime("%Y%m%d")

            df = self.market_data.get_index_daily_prices("0001", start_date, end_date)
            if df.empty or len(df) < 20:
                self._bear_score = 0
                self._bear_market = False
                return

            closes = pd.to_numeric(df["bstp_nmix_prpr"], errors="coerce").dropna()
            if len(closes) < 20:
                self._bear_score = 0
                self._bear_market = False
                return

            score = 0

            # 1. KOSPI < MA20: 중기 하락 추세
            ma20 = closes.tail(20).mean()
            current = closes.iloc[-1]
            if current < ma20:
                score += 1

            # 2. MA5 < MA20: 단기 데드크로스
            ma5 = closes.tail(5).mean()
            if ma5 < ma20:
                score += 1

            # 3. 3일 연속 하락
            if len(closes) >= 4:
                last3 = closes.iloc[-3:]
                prev3 = closes.iloc[-4:-1]
                if all(c < p for c, p in zip(last3, prev3)):
                    score += 1

            self._bear_score = score
            self._bear_market = score >= 1

            logger.info("시장 레짐: 약세점수=%d (KOSPI: %.1f, MA20: %.1f, MA5: %.1f)",
                        score, current, ma20, ma5)
        except Exception as e:
            logger.warning("시장 레짐 확인 실패: %s", e)
            self._bear_score = 0
            self._bear_market = False

    def _estimate_market_from_quotes(self, quotes: List[Quote]):
        """배치 시세에서 약세 점수를 추정한다 (백테스트용).

        인버스 ETF를 제외한 일반 종목의 등락률로 판단.
        """
        regular_quotes = [q for q in quotes if q.symbol not in self._inverse_symbols]
        if not regular_quotes:
            return

        total = len(regular_quotes)
        avg_change = sum(q.change_rate for q in regular_quotes) / total
        declining = sum(1 for q in regular_quotes if q.change_rate < 0)
        decline_ratio = declining / total

        score = 0

        # 평균 등락률 < -0.5%
        if avg_change < -0.5:
            score += 1
        # 평균 등락률 < -1.0%
        if avg_change < -1.0:
            score += 1
        # 하락 종목 비율 > 70%
        if decline_ratio > 0.7:
            score += 1

        self._bear_score = score
        self._bear_market = score >= 1

    def _evaluate_buy(self, quote: Quote, pending_orders: Optional[List[Order]] = None) -> Optional[Order]:
        """모멘텀 점수 기반 매수 판단 (일반 주식)."""
        # 인버스 ETF는 별도 로직
        if quote.symbol in self._inverse_symbols:
            return None

        if quote.current_price <= 0 or quote.open_price <= 0:
            return None
        if quote.current_price < self.cfg.min_price:
            return None

        # 쿨다운 체크
        last_sold = self._sell_cooldown.get(quote.symbol)
        if last_sold:
            elapsed = (datetime.now() - last_sold).total_seconds()
            if elapsed < self.cfg.cooldown_seconds:
                return None

        position = self.positions.get(quote.symbol)
        is_scale_in = position is not None

        # 약세장 보수 모드(B): 신규 롱 진입 금지
        if not is_scale_in and self._bear_market and self.cfg.bear_market_mode == 'B':
            return None

        score = self._calc_momentum_score(quote)
        if is_scale_in:
            if not self.cfg.enable_pyramiding:
                return None
            pnl_pct = (quote.current_price - position.buy_price) / position.buy_price * 100
            if pnl_pct < self.cfg.scale_in_min_profit_pct:
                return None
            if score < (self.cfg.min_momentum_score + self.cfg.scale_in_score_bonus):
                return None
        else:
            if score < self.cfg.min_momentum_score:
                return None

        alloc = self._compute_buy_allocation(
            symbol=quote.symbol,
            current_price=quote.current_price,
            pending_orders=pending_orders,
        )
        quantity = alloc // quote.current_price

        if quantity <= 0:
            return None

        if is_scale_in:
            logger.info(
                "추가매수 신호: %s(%s) 점수=%.1f, %d주 @ %s원 (할당 %s원)",
                quote.name,
                quote.symbol,
                score,
                quantity,
                f"{quote.current_price:,}",
                f"{alloc:,}",
            )
        else:
            logger.info(
                "매수 신호: %s(%s) 점수=%.1f, %d주 @ %s원 (할당 %s원)",
                quote.name,
                quote.symbol,
                score,
                quantity,
                f"{quote.current_price:,}",
                f"{alloc:,}",
            )

        return Order(
            symbol=quote.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=0,
        )

    def _evaluate_inverse_buy(self, quote: Quote, pending_orders: Optional[List[Order]] = None) -> Optional[Order]:
        """인버스 ETF 매수 판단.

        약세 점수 >= bearish_threshold일 때만 진입.
        인버스 ETF는 시장 하락 시 상승하므로 모멘텀 점수가 자연스럽게 높아진다.
        """
        if quote.current_price <= 0 or quote.open_price <= 0:
            return None

        if self._bear_score < self.cfg.bearish_threshold:
            return None

        # 쿨다운 체크
        last_sold = self._sell_cooldown.get(quote.symbol)
        if last_sold:
            elapsed = (datetime.now() - last_sold).total_seconds()
            if elapsed < self.cfg.cooldown_seconds:
                return None

        # 모멘텀 점수 (인버스도 상승 중이어야 진입)
        score = self._calc_momentum_score(quote)
        if score < self.cfg.inverse_min_momentum:
            return None

        alloc = self._compute_buy_allocation(
            symbol=quote.symbol,
            current_price=quote.current_price,
            pending_orders=pending_orders,
        )
        quantity = alloc // quote.current_price
        if quantity <= 0:
            return None

        logger.info("[INV] 매수 신호: %s 약세점수=%d, 모멘텀=%.1f, %d주 @ %s원",
                     quote.symbol, self._bear_score, score, quantity,
                     f"{quote.current_price:,}")

        return Order(
            symbol=quote.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=0,
        )

    def _compute_buy_allocation(
        self,
        symbol: str,
        current_price: int,
        pending_orders: Optional[List[Order]] = None,
    ) -> int:
        total_exposure = self._get_total_exposure()
        stock_exposure = self._get_stock_exposure(symbol)

        pending_total = 0
        pending_stock = 0
        for order in pending_orders or []:
            if order.side != OrderSide.BUY:
                continue
            q = self._quotes_cache.get(order.symbol)
            if q is None or q.current_price <= 0:
                continue
            amount = q.current_price * order.quantity
            pending_total += amount
            if order.symbol == symbol:
                pending_stock += amount

        total_room = self.cfg.seed_money - (total_exposure + pending_total)
        stock_room = self.cfg.max_per_stock_amount - (stock_exposure + pending_stock)
        alloc = min(self.cfg.per_stock_amount, total_room, stock_room)
        if alloc <= 0 or current_price <= 0:
            return 0
        return alloc

    def _get_total_exposure(self) -> int:
        return sum(pos.buy_price * pos.quantity for pos in self.positions.values())

    def _get_stock_exposure(self, symbol: str) -> int:
        pos = self.positions.get(symbol)
        if pos is None:
            return 0
        return pos.buy_price * pos.quantity

    def _evaluate_sell(self, quote: Quote) -> Optional[Order]:
        """익절/손절/추적손절 판단 (일반 주식)."""
        pos = self.positions.get(quote.symbol)
        if not pos:
            return None

        pnl_pct = (quote.current_price - pos.buy_price) / pos.buy_price * 100
        pnl_amount = (quote.current_price - pos.buy_price) * pos.quantity

        # 익절
        if pnl_pct >= self.cfg.take_profit_pct:
            logger.info("익절: %s %.2f%% (%s원)",
                        quote.symbol, pnl_pct, f"{pnl_amount:,}")
            return self._make_sell_order(pos)

        # 개별 포지션 손절 (금액 기준)
        if pnl_amount <= self.cfg.per_position_stop_loss:
            logger.info("개별손절: %s %s원 (한도 %s원)",
                        quote.symbol, f"{pnl_amount:,}",
                        f"{self.cfg.per_position_stop_loss:,}")
            return self._make_sell_order(pos)

        # 추적손절 (고점 대비)
        if pos.high_since_buy > pos.buy_price:
            drop_from_high = (quote.current_price - pos.high_since_buy) / pos.high_since_buy * 100
            if drop_from_high <= self.cfg.trailing_stop_pct:
                logger.info("추적손절: %s 고점 %s → 현재 %s (%.2f%%)",
                            quote.symbol, f"{pos.high_since_buy:,}",
                            f"{quote.current_price:,}", drop_from_high)
                return self._make_sell_order(pos)

        return None

    def _evaluate_inverse_sell(self, quote: Quote) -> Optional[Order]:
        """인버스 ETF 매도 판단 (타이트한 리스크 관리).

        인버스 ETF는 음의 복리 위험이 있으므로:
        - 익절/손절 기준이 일반보다 타이트
        - 시간 기반 강제 청산 (최대 2시간)
        - 시장 반등 시(약세 점수 하락) 즉시 청산
        """
        pos = self.positions.get(quote.symbol)
        if not pos:
            return None

        pnl_pct = (quote.current_price - pos.buy_price) / pos.buy_price * 100

        # 1. 익절 (+1.0%, 일반 +1.5%보다 빠르게)
        if pnl_pct >= self.cfg.inverse_take_profit_pct:
            logger.info("[INV] 익절: %s %.2f%%", quote.symbol, pnl_pct)
            return self._make_sell_order(pos)

        # 2. 손절 (-0.5%, 타이트)
        if pnl_pct <= self.cfg.inverse_stop_loss_pct:
            logger.info("[INV] 손절: %s %.2f%%", quote.symbol, pnl_pct)
            return self._make_sell_order(pos)

        # 3. 시간 초과 청산 (음의 복리 방지, 실거래 모드만)
        if self.market_data is not None:
            hold_minutes = (datetime.now() - pos.buy_time).total_seconds() / 60
            if hold_minutes >= self.cfg.inverse_max_hold_minutes:
                logger.info("[INV] 시간초과 청산: %s (%.0f분 보유)", quote.symbol, hold_minutes)
                return self._make_sell_order(pos)

        # 4. 시장 반등 청산 (약세 점수가 임계 미만으로 떨어지면)
        if self._bear_score < self.cfg.bearish_threshold:
            logger.info("[INV] 시장반등 청산: %s (약세점수: %d)", quote.symbol, self._bear_score)
            return self._make_sell_order(pos)

        # 5. 추적손절 (고점 -0.3%, 일반 -0.7%보다 타이트)
        if pos.high_since_buy > pos.buy_price:
            drop_from_high = (quote.current_price - pos.high_since_buy) / pos.high_since_buy * 100
            if drop_from_high <= self.cfg.inverse_trailing_stop_pct:
                logger.info("[INV] 추적손절: %s 고점 %s → 현재 %s (%.2f%%)",
                            quote.symbol, f"{pos.high_since_buy:,}",
                            f"{quote.current_price:,}", drop_from_high)
                return self._make_sell_order(pos)

        return None

    def _calc_momentum_score(self, quote: Quote) -> float:
        """모멘텀 점수를 계산한다 (0~5)."""
        score = 0.0

        # 1. 시가 대비 상승폭 (0~1.5)
        if quote.open_price > 0:
            vs_open = (quote.current_price - quote.open_price) / quote.open_price * 100
            if vs_open >= 2.0:
                score += 1.5
            elif vs_open >= 1.0:
                score += 1.0
            elif vs_open >= 0.5:
                score += 0.5

        # 2. 전일 대비 등락률 (0~1.0)
        if quote.change_rate >= 2.0:
            score += 1.0
        elif quote.change_rate >= 1.0:
            score += 0.6
        elif quote.change_rate >= 0.5:
            score += 0.3

        # 3. 고가 근접도 (0~1.0)
        price_range = quote.high_price - quote.low_price
        if price_range > 0:
            proximity = (quote.current_price - quote.low_price) / price_range
            if proximity >= 0.9:
                score += 1.0
            elif proximity >= 0.7:
                score += 0.5

        # 4. 거래량 폭발 (0~1.5) — 5일 평균 대비
        avg_vol = self._avg_volumes.get(quote.symbol)
        if avg_vol and avg_vol > 0:
            vol_ratio = quote.volume / avg_vol
            if vol_ratio >= 3.0:
                score += 1.5
            elif vol_ratio >= 2.0:
                score += 1.0
            elif vol_ratio >= 1.5:
                score += 0.5

        return score

    def _make_sell_order(self, pos: PositionState) -> Order:
        return Order(
            symbol=pos.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=pos.quantity,
            price=0,
        )

    def _calc_commission_cost(self, notional: int) -> int:
        if notional <= 0:
            return 0
        return int(round(notional * self.cfg.commission_rate))

    def _calc_sell_tax_slippage_cost(self, sell_notional: int) -> int:
        if sell_notional <= 0:
            return 0
        return int(round(sell_notional * self.cfg.tax_slippage_rate))

    def _estimate_unrealized_net_pnl(self) -> int:
        total = 0
        for sym, pos in self.positions.items():
            q = self._quotes_cache.get(sym)
            if not q or q.current_price <= 0:
                continue
            gross = (q.current_price - pos.buy_price) * pos.quantity
            sell_notional = q.current_price * pos.quantity
            exit_cost = (
                self._calc_commission_cost(sell_notional) +
                self._calc_sell_tax_slippage_cost(sell_notional)
            )
            total += (gross - exit_cost)
        return total

    def _liquidate_all(self) -> List[Order]:
        """전 포지션 청산 (일반 + 인버스 모두)."""
        orders = []
        for pos in self.positions.values():
            orders.append(self._make_sell_order(pos))
        return orders

    def load_avg_volumes(self, avg_volumes: Dict[str, int]):
        """5일 평균 거래량을 외부에서 주입한다 (백테스트/초기화 시)."""
        self._avg_volumes = avg_volumes
