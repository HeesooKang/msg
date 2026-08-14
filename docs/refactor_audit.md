# 단일 EV 런타임 감사 기록

최종 갱신일: 2026-08-10

## 유일한 라이브 경로

```text
bot_ctl.sh
-> run_bot.py
-> src.main.run_scheduled()
-> TradingScheduler._run_trading_session()
-> MomentumScalpStrategy.on_batch_tick()
```

`run_bot.py`가 KIS 클라이언트와 전략을 한 번 생성합니다. `TradingScheduler`는 전략의 동일한 `market_data.client`를 직접 사용하며 별도 클라이언트 생성, 다른 전략 fallback, 존재하지 않는 동적 훅을 사용하지 않습니다.

틱의 판단 순서는 아래 하나입니다.

1. WebSocket 최신 시세와 확정 주문 결과 정산
2. 보유 포지션의 하드스톱 또는 선택된 만기 청산
3. 모든 최신·구매 가능 종목의 특성을 한 번 계산
4. 확정된 30·60·120·180초 순수익을 동일 ridge 공식으로 예측
5. 종목·만기별 `_build_trade_plan()` 생성
6. 비용 포함 `expected_net` 최고 계획 한 건 주문
7. 실제 매수 체결 뒤 한 번 재검증

## 남긴 판단값

- 가격·실제 체결량·호가 잔량·상대수익률·spread 특성
- 만기별 ridge 예상 순수익률과 하단 순수익률
- 원화 `expected_net`, `lower_net`, `committed_risk_net_abs`
- 자본 한도, 남은 일일 손실 여유, 선택된 만료시각

시장 레짐, breadth, leader percentile, micro score, 거래량 방향 추정, confidence, 승률 gate, 인버스 전용 판단은 없습니다. 인버스 ETF도 일반 종목과 같은 함수를 통과합니다.

## 삭제한 런타임 코드

- opening/intraday/inverse별 route, gate, override, rescue, focus 경로
- `regime_router`, `math_signals`, `quote_tape`, `momentum_scalp_micro`
- 분리돼 있던 레거시 entry/conviction/exit/fill/state 전략 모듈
- WebSocket 전환 뒤 호출되지 않던 예측용 REST 시세 캐시와 캐시 나이 조회
- 호출되지 않던 시가총액 순위 fallback과 `RankingItem`
- 전략에서 한 번도 읽지 않던 shortlist, prediction-call, session-start 상태
- 존재하지 않는 `_has_unconfirmed_daily_breaker_sell_fills` 동적 훅
- 스케줄러의 별도 KIS 클라이언트 생성 fallback
- `run_scheduled()`와 스케줄러에 중복 전달되던 두 번째 runtime config 경로
- 실제 타입 대신 구형 mock/객체를 허용하던 `getattr`/`callable` 실행 fallback
- `src/main.py`의 독립 연결 테스트 진입점과 숨은 `_runtime_config`
- 적용되지 않던 HTTP transport cooldown 상태
- 주문 검증기에서 읽히지 않던 `max_position_count`, `_daily_loss` 상태
- 예측기가 읽지 않던 Quote 호가량·거래량·OHLC·체결방향 필드
- 계좌 동기화가 읽지 않던 Position 평가 필드

별도 백테스트 실행에 필요한 일봉·분봉 조회와 로그 로테이션 프레임워크 훅은 라이브 경로 밖의 명시적 실행 기능이므로 유지합니다.

## 유지한 운영 안전 코드

- 주문 체결 확인, 부분체결, 미확정 주문 중복 방지
- 계좌 포지션 동기화와 재시작 상태 복구
- KIS 호출 제한 냉각과 WebSocket 재연결
- 보유종목 WebSocket stale 시 REST 비상 조회
- 확정 매도체결 원장 기반 당일 손익
- `+10,000원` 신규 진입 종료와 실현+미실현 `-5,000원` 하드스톱
- 사용자가 명시적으로 켜는 당일 하드스톱 재시작 기준점
- 실제 계좌 전환 readiness 제한

마지막 항목은 종목 진입 판단 gate가 아니라 `TRADING_MODE=real`에 실제로 쓰이는 계좌 안전 기능입니다.

## 현재 규모

- `src/strategies/momentum_scalp.py`: 2,093줄, 메서드 72개
- `src/scheduler.py`: 1,328줄, 메서드 50개
- `src/analytics/price_prediction.py`: 533줄
- 전략·예측·시세·스케줄러 핵심 파일 합계: 4,359줄

정적 감사에서 라이브 소스에 정의만 남은 함수는 없었습니다. 참조 1회로 보이는 항목은 외부 프레임워크가 이름으로 호출하는 로그 로테이션 훅과 별도 백테스트 데이터 조회뿐입니다.

## 정리 원칙

- 새 route/gate/override/rescue/filter를 추가하지 않습니다.
- 삭제한 코드를 다른 파일로 옮겨 숨기지 않습니다.
- 예측 정확도는 forecast 원장의 30·60·120·180초 실행가격 결과로 검증합니다.
- 체결·손익·복구·API 안전 코드는 거래 판단과 분리해서 유지합니다.
- 재시작은 사용자가 직접 수행합니다.
