from abc import ABC, abstractmethod
from typing import List

from src.models import Order, OrderResult, Quote


class BaseStrategy(ABC):
    """매매 전략의 추상 베이스 클래스.

    새로운 전략을 만들려면 이 클래스를 상속하고
    initialize(), on_tick(), should_continue()를 구현하면 된다.
    """

    @abstractmethod
    def initialize(self):
        """전략 초기화. 봇 시작 시 1회 호출된다."""
        pass

    @abstractmethod
    def get_watchlist(self) -> List[str]:
        """감시할 종목코드 리스트를 반환한다."""
        pass

    @abstractmethod
    def on_tick(self, quote: Quote) -> List[Order]:
        """시세 데이터를 받아 주문 리스트를 반환한다.

        주문할 게 없으면 빈 리스트를 반환한다.
        """
        pass

    def on_batch_tick(self, quotes: List[Quote]) -> List[Order]:
        """여러 종목의 시세를 한번에 받아 주문 리스트를 반환한다.

        기본 구현은 on_tick()을 개별 호출한다. 오버라이드 가능.
        """
        orders = []
        for q in quotes:
            orders.extend(self.on_tick(q))
        return orders

    def on_order_filled(self, result: OrderResult):
        """주문이 체결됐을 때 호출된다. 오버라이드 가능."""
        pass

    @abstractmethod
    def should_continue(self) -> bool:
        """False를 반환하면 봇이 종료된다."""
        pass
