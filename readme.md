# 주식 자동거래 프로그램 (`msg`)

한국투자증권 KIS Open API를 사용하는 국내주식 자동거래 봇입니다. 현재 운영 전략은 시장 레짐이나 장세별 라우터 없이 모든 감시 종목을 같은 실행가격 예측·EV 공식으로 평가합니다.

## 실행 경로

```text
bot_ctl.sh
-> run_bot.py
-> src.main.run_scheduled()
-> TradingScheduler
-> MomentumScalpStrategy.on_batch_tick()
```

`run_bot.py`가 설정, KIS 클라이언트, 시세 API와 전략을 한 번만 생성합니다. 스케줄러는 전략이 가진 동일 클라이언트를 사용하며 별도 전략·클라이언트 fallback을 만들지 않습니다.

## 장중 흐름

한 틱의 순서는 다음과 같습니다.

1. WebSocket 체결 시세를 1초 단위 최신 호가로 병합합니다.
2. 확정 주문과 계좌 포지션을 정산합니다.
3. 보유 포지션의 하드스톱 또는 선택된 예측 만료 청산을 판단합니다.
4. 구매 가능한 모든 최신 종목의 특성을 한 번 계산하고 30·60·120·180초 실행 순손익을 같은 ridge 공식으로 평가합니다.
5. 종목과 만기를 합친 후보 중 비용 포함 `expected_net`이 가장 큰 양수 EV 계획 하나만 지정가 매수합니다.
6. 체결 가격으로 EV를 한 번 재검증한 뒤 계획을 유지하거나 즉시 청산합니다.

예측 입력은 한 번만 계산되는 아래 실행 시세 특성입니다.

- 15·60·180초 수익률, 60초 고점 대비 되돌림과 실현 변동성
- 거래량 증가, 실제 누적 매수·매도 체결량 차이, 호가 잔량 불균형
- 감시풀 중앙값 대비 60초 상대수익률과 현재 spread

각 만기의 확정된 `ask` 진입·`bid` 청산 비용 차감 순수익을 동일한 NumPy ridge 공식으로 학습합니다. 시간대 분기, 레짐, breadth, leader percentile, 별도 인버스 라우트, confidence gate는 사용하지 않습니다.

## 감시 종목

- 고정 감시 32종목
- 인버스 ETF 4종목
- KIS 등락률 순위에서 찾은 동적 종목 30개
- 현재 보유 종목

인버스 ETF도 일반 종목과 같은 예측·EV·수량 공식을 사용합니다. 순위 API는 감시 종목 발견에만 쓰며 진입 자격이나 점수를 만들지 않습니다.

## 손익과 청산

- 일일 확정 실현손익 목표: `+10,000원`
- 실현+미실현 손실 하드스톱: `-5,000원`
- 한 거래가 남은 목표 전부를 채울 필요는 없습니다.
- 수량은 자본 한도와 남은 일일 손실 여유 안에서 계산합니다.
- 일반 청산은 EV가 선택한 30·60·120·180초 만료 시장가입니다.
- 주문 체결, 부분체결, 재시작 복구, API 호출 제한 처리는 운영 필수 경로로 유지합니다.

`ALLOW_DAILY_HARD_STOP_BYPASS=true`로 시작하면 이미 당일 손실한도에 도달한 날의 재시작 시점 손익을 새 손실 기준점으로 삼습니다. 이 설정은 사용자가 명시적으로 켠 당일에만 적용됩니다.

## 주요 파일

```text
run_bot.py                              봇 조립과 실행
src/main.py                             스케줄러 시작
src/scheduler.py                        장 시간, 시세, 주문, 체결 정산
src/market_stream.py                    KIS WebSocket 시세
src/market_data.py                      동적 종목 발견과 비상 REST 조회
src/analytics/price_prediction.py       단일 특성 계산과 다중 만기 ridge 예측
src/analytics/forecast_outcomes.py      다중 만기 실행가격 결과 원장
src/strategies/momentum_scalp.py        단일 EV 전략
src/strategies/momentum_scalp_types.py  설정·계획·포지션 상태
src/strategies/momentum_scalp_pnl.py    원화 비용·순손익 계산
src/performance_reporting.py            일일 성적표와 실계좌 준비도
```

`src/backtest/`와 `run_backtest*.py`는 별도 검증 도구이며 라이브 봇의 장중 호출 경로에는 포함되지 않습니다.

## 실행

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./bot_ctl.sh restart
```

운영 명령:

```bash
./bot_ctl.sh status
./bot_ctl.sh today
./bot_ctl.sh report
./bot_ctl.sh gate
./bot_ctl.sh logs
./bot_ctl.sh monitor
```

`gate`는 종목 진입 gate가 아니라 실제 계좌 전환을 제한하는 운영 안전 기능입니다. `paper` 성과가 준비 기준을 통과하지 못하면 `real` 실행을 막고, 통과 뒤에도 자본과 손실한도를 단계적으로 확대합니다.

## 로그와 상태

- `logs/trading.log`: 배치 평가, 선택 후보, 계획 청산, 체결 정산, 세션 상태
- `logs/orders.log`: 실제 주문 제출 결과와 확정 체결 가격
- `reports/forecast-outcomes/`: 동일 신호의 30·60·120·180초 실행가격 결과
- `reports/YYYY/MM/daily-scorecard.*`: 일일 확정 손익 성적표
- `reports/real-trade-readiness.*`: 실제 계좌 전환 준비도
- `state/momentum_scalp_daily_state.json`: 당일 포지션·계획·확정 체결 원장 복구 상태

로그, 리포트, 런타임 상태와 토큰 파일은 Git 추적 대상이 아닙니다.

## 검증

```bash
venv/bin/python -m unittest discover tests
```

핵심 계약은 실행가격 특성의 미래 데이터 차단, 만기별 비용 포함 EV, 고가 1주 구매 불가, 남은 손실 여유 수량, 일반주·인버스 동일 공식, 선택된 만기 청산, 부분체결 정산, 일일 목표와 하드스톱입니다.
