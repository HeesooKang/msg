# EV 리팩터링 맥락 재고정용 마스터 프롬프트

이 문서는 새 세션 시작 시 그대로 붙여넣거나, 작업 중 에이전트가 기존 큰 줄기에서 벗어날 때 다시 붙여넣기 위한 작업 헌장입니다.

## 복붙용 프롬프트

```text
당신은 현재 작업 중인 `msg` 저장소의 트레이딩 봇 리팩터링을 맡은 코딩 에이전트입니다. 반드시 존댓말을 사용하십시오.

이번 작업의 절대 목표는 “로직 간소화”입니다. 기존처럼 gate, override, rescue, filter, focus, bypass를 계속 추가해서 문제를 덮지 마십시오. 지금 코드는 진입 로직이 중구난방으로 비대해져 있고, 작은 문제를 고칠 때마다 새 예외가 붙어서 전체 판단 구조가 망가졌습니다. 이번 작업은 새 조건을 덧붙이는 작업이 아니라, 롱 진입 판단을 기댓값 중심으로 재구성하는 작업입니다.

핵심 목표:
- 하루 수익 목표는 순수익 +10,000원입니다.
- 하루 손실 제한은 -5,000원입니다.
- 한 번의 거래로 무조건 10,000원을 벌어야 한다는 뜻이 아닙니다.
- 예측상 1~2% 상승 가능성이 있고 비용 반영 후 양수 EV라면, 2,000원/3,000원 기대수익 거래도 허용해서 누적으로 목표에 접근해야 합니다.
- 손실 제한은 엄격하게 유지하십시오. 남은 일일 손실 여유를 넘기는 수량은 절대 선택하지 마십시오.

절대 금지:
- 새 gate/override/rescue/filter를 추가하지 마십시오.
- 기존 `daily_target_focus`, `expected_target_net`, `expected_rr_ratio`, `risk_sizing_no_valid_quantity`, `price_prediction_*_override` 계열을 살리려고 또 우회 패치를 하지 마십시오.
- 문제를 한 함수 안에서 더 복잡한 if문으로 덮지 마십시오.
- 컨텍스트가 압축되거나 작업이 길어져도, 임의로 네트워크 timeout, 로그 포맷, 문서 정리 같은 옆길로 빠지지 마십시오.
- 사용자가 직접 한다고 한 재시작을 수행하지 마십시오.

먼저 반드시 읽을 것:
- `AGENTS.md`
- `docs/refactor_audit.md`
- `src/strategies/momentum_scalp.py`
- `src/strategies/momentum_scalp_types.py`
- `src/strategies/momentum_scalp_exit.py`
- `src/strategies/momentum_scalp_pnl.py`
- `src/analytics/price_prediction.py`
- `src/analytics/forecast_outcomes.py`
- 관련 테스트: `tests/test_risk_controls_unittest.py`, `tests/test_momentum_scalp_refactor_unittest.py`

현재 런타임에서 import하지 않는 `momentum_scalp_entry.py`, `momentum_scalp_entry_filters.py`, `momentum_scalp_entry_overrides.py`, `momentum_scalp_conviction.py`의 레거시 경로를 다시 연결하지 마십시오.

읽은 뒤 먼저 요약하십시오:
1. 현재 롱 진입 호출 흐름
2. 제거할 gate/override/filter 경로
3. 유지할 안전장치
4. 새 단일 EV plan 함수의 입출력
5. 테스트 계획

목표 구조:
- 후보 생성/랭킹은 유지합니다.
- 최종 롱 매수 판단은 단일 함수로 통합합니다.
  - 예: `_build_expected_value_trade_plan(quote, context, pending_orders)`
- 이 함수가 수량, 기대 순손익, 예측 순손익, 하단 순손익, 승률, 목표 순익, 손절 순손실을 한 번에 결정합니다.
- EV 계산은 비용 포함 순손익 기준으로 통일합니다.
- 수량은 1주부터 가능한 최대 수량까지 평가해서, 남은 일일 손실 여유 안에서 EV가 가장 큰 수량을 선택합니다.
- `expected_net > 0`, `predicted_net > 0`, `prediction ready`, `planned_stop_net <= remaining_daily_loss_room`이면 진입을 허용합니다.
- `min_expected_net_profit`이나 “남은 일일 목표를 한 번에 채워야 함” 같은 기준으로 진입을 막지 마십시오.
- 청산은 `planned_target_net_pnl`과 `planned_stop_net_loss_abs`를 우선합니다.
- 손절성 청산은 반등 예측 가드 없이 즉시 실행합니다.
- 인버스 ETF를 위한 별도 gate/route/switch를 만들지 말고 일반 종목과 같은 예측/EV 경로로 평가하십시오.

완료 기준:
- 롱 라이브 진입 경로에서 기존 복잡한 gate 체인이 최종 판단을 하지 않습니다.
- 양수 EV의 작은 기대수익 거래가 허용됩니다.
- 하루 손실 제한 -5,000원을 넘길 수 있는 거래는 수량 축소 또는 거부됩니다.
- 진입 메타와 포지션 상태에 planned target/stop/EV가 기록됩니다.
- 관련 단위 테스트를 추가/수정하고, `venv/bin/python -m unittest discover tests`가 통과해야 합니다.

작업 중 길을 잃었다고 판단되면 즉시 이 프롬프트의 “핵심 목표 / 절대 금지 / 목표 구조 / 완료 기준”으로 되돌아가십시오.
```

## 사용 규칙

- 새 세션 시작 시 위 코드블록 전체를 첫 메시지로 붙여넣습니다.
- 에이전트가 timeout, 로그 포맷, 미세 threshold, 기존 override 보수 같은 옆길로 빠지면 같은 프롬프트를 다시 붙여넣습니다.
- 구현 중간에도 “현재 변경이 위 프롬프트의 목표 구조를 단순화하고 있는지 먼저 점검해라”라고 붙이면 복귀 기준으로 사용합니다.

## 범위

- 최종 목적은 기존 로직을 더 정교하게 만드는 것이 아니라, 판단 경로를 줄이는 것입니다.
- 이 프롬프트는 롱 진입/청산 간소화에 초점을 둡니다.
- 인버스, 카카오 알림, API timeout, 문서 정리, 실계좌 readiness gate는 이 프롬프트의 기본 범위 밖입니다.
