# MomentumScalp 리팩터링 감사 기록

최종 갱신일: 2026-07-21

## 런타임 계약

- 시작 흐름은 `bot_ctl.sh -> run_bot.py -> src.main.run_scheduled -> TradingScheduler`입니다.
- 스케줄러 핵심 훅은 `initialize()`, `get_watchlist()`, `on_batch_tick()`, `on_order_filled()`, `should_continue()`입니다.
- 상태 및 체결 복구 훅인 `update_runtime_pool`, `has_runtime_state_snapshot`, `sync_positions_from_account`, `reconcile_pending_fills_from_account`, `reconcile_no_holding_sell_failures_from_account`, `confirm_reconciled_sell_fills`도 유지합니다.
- 사용자가 직접 수행한다고 한 봇 재시작은 코드 작업 과정에서 실행하지 않습니다.

## 현재 진입 흐름

1. `on_batch_tick()`이 최신 시세와 180초 예측 결과 원장을 갱신합니다.
2. `_update_market_state()`가 후보 풀의 리더 신호와 동적 후보 큐를 갱신합니다.
3. `_long_entry_shortlist()`가 현재 평가 가능한 모든 후보를 만듭니다.
4. 각 후보에 `_build_expected_value_candidate()`와 `_build_expected_value_trade_plan()`을 동일하게 적용합니다.
5. 비용 반영 기대 순손익, 예측 순손익, 하단 순손익, 승률, 손절 위험과 남은 일일 손실 여유를 한 계획에서 계산합니다.
6. `_rank_expected_value_candidates()`가 전체 후보를 EV 순으로 정렬하고, 허용된 최고 EV 후보 하나만 주문으로 만듭니다.
7. 인버스 ETF도 별도 라우트나 스위치 없이 동일한 후보/예측/EV 경로를 사용합니다.

## 유지 안전장치

- 유효하지 않은 종목 코드 및 1주 가격이 예산을 초과하는 종목 제외
- 중복·미확정 매수, 보유 수와 총 노출 한도, API 냉각, 종목 주문 불가 상태 확인
- 열린 포지션과 미확정 주문의 계획 손실까지 합산한 일일 손실 여유 계산
- 비용 포함 `expected_net > 0`, `predicted_net > 0`, 준비된 예측, 손실 여유 내 계획 위험
- `planned_target_net_pnl`과 `planned_stop_net_loss_abs` 우선 청산
- 손절 주문을 반등 예측으로 미루지 않는 즉시 청산

## 제거한 경로

- 기존 opening/intraday conviction 게이트 체인과 우회·구제·override·focus 경로
- 과거 EV가 라이브 예측 전에 진입을 막거나 살리는 경로
- `min_expected_net_profit` 또는 남은 일일 목표를 한 거래에서 채우도록 강제하던 경로
- 별도의 인버스 진입 라우트와 인버스 enable 스위치
- 전략 전체 손실 쿨다운 및 관련 상태 저장
- 서로 다른 후보 랭킹과 장문의 레거시 후보 디버그 로그
- 실제로 `guardable=True`가 생성되지 않아 실행될 수 없던 청산 반등 보류 로직과 설정·상태
- 삭제된 private 구현만 직접 호출하던 단위 테스트

런타임과 테스트에서 import되지 않던 `momentum_scalp_entry.py`, `momentum_scalp_entry_filters.py`, `momentum_scalp_entry_overrides.py`, `momentum_scalp_conviction.py` 레거시 파일은 제거했습니다.

## 파일 규모

- `src/strategies/momentum_scalp.py`: 이번 정리 전 11,366줄, 현재 4,548줄
- 본체 메서드: 현재 143개이며, `src/` 기준 런타임 루트에서 모두 도달 가능함을 AST 호출 그래프로 확인했습니다.
- 별도 유지 모듈: 체결 `momentum_scalp_fills.py`, 청산 결정 `momentum_scalp_exit.py`, 손익 수학 `momentum_scalp_pnl.py`, 상태 집계 `momentum_scalp_state.py`, 가격 예측 `price_prediction.py`, 예측 결과 원장 `forecast_outcomes.py`

## 검증

- 영향 파일 `py_compile` 통과
- `venv/bin/python -m unittest discover tests`: 248개 통과
- 단일 EV 계획의 작은 양수 기대수익 허용, 열린 포지션 위험 합산, 인버스 동일 경로, 계획 손절 즉시 실행 테스트를 유지합니다.

## 다음 정리 원칙

- 줄 수만 줄이기 위한 파일 이동은 하지 않습니다.
- 현재 런타임 호출이 없는 코드부터 제거합니다.
- 새 gate/override/rescue/filter를 만들지 않습니다.
- 예측 정확도 개선은 `price_prediction.py`와 예측 결과 원장의 walk-forward 보정에서 수행하고, 주문 경로에는 별도 예외 체인을 추가하지 않습니다.
