from abc import ABC, abstractmethod
from typing import List

from src.models import Order, OrderResult, Quote


class BaseStrategy(ABC):
    """매매 전략의 추상 베이스 클래스.

    전략은 배치 시세를 유일한 실행 입력으로 사용한다.
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
    def on_batch_tick(self, quotes: List[Quote]) -> List[Order]:
        """현재 배치 전체를 한 번 평가해 주문 리스트를 반환한다."""
        pass

    @abstractmethod
    def on_order_filled(self, result: OrderResult):
        """확정되거나 재조정이 필요한 주문 결과를 반영한다."""
        pass

    @abstractmethod
    def should_continue(self) -> bool:
        """False를 반환하면 봇이 종료된다."""
        pass
