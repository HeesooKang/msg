# 주식 자동거래 프로그램 (`msg`)

한국투자증권 KIS Open API를 사용하는 주식 자동거래 프로젝트입니다.  
현재 기준 운영 철학은 `paper 모의투자 우선`, `실계좌는 게이트 통과 후 단계적 전환`입니다.

## 현재 상태 요약

- 실시간 자동매매 봇: `run_bot.py`
- 일봉 smoke 백테스트: `run_backtest.py`
- 1분봉 리플레이 백테스트: `run_backtest_intraday.py`
- 핵심 전략: `src/strategies/momentum_scalp.py`
- 레짐 라우터: `src/strategies/regime_router.py`
- 일일 성적표 / 실투자 준비도 리포트: `src/performance_reporting.py`

이 프로젝트는 더 이상 단순 모멘텀 점수 하나로만 진입하지 않습니다.  
현재 구조는 `레짐 판별 -> 레짐별 서브전략 선택 -> 진입/청산/리스크 제어 -> 리포트 생성` 흐름입니다.

## 전략 구조

현재 진입 로직은 `레짐 라우터 + 장세별 서브전략` 구조입니다.

- `bull_breakout_strategy`
강세장에서만 롱을 봅니다. 최근 고점 돌파, 거래량 스파이크, 확인 틱 유지가 핵심입니다.

- `neutral_pullback_strategy`
중립장에서만 롱을 봅니다. `상승 -> 눌림 -> 재돌파` 형태만 허용하고, 고점 추격은 차단합니다.

- `soft_bear_defense_strategy`
완만약세용 인버스 방어 전략 경로입니다. 코드상 존재하지만, 현재 기본 운영값에서는 완만약세 신규 진입을 매우 보수적으로 둡니다.

- `hard_bear_inverse_strategy`
강한 약세장에서 인버스 ETF만 봅니다.

레짐은 대략 아래 4개로 나뉩니다.

- `bull`
- `neutral`
- `soft_bear`
- `bear`

## 현재 리스크 제어

기본 운영값은 손실 억제를 매우 강하게 둡니다.

- 일일 목표: `+10,000원`
- 일일 하드스탑: `-5,000원`
- 손실 1단계: `-2,000원`
- 중립장 손실 후: `30분 쿨다운 + A급 재도전 1회만 허용`
- 장마감 직전 신규 진입 차단: `15:00-15:21`

중요한 점은, `실계좌` 모드라고 해서 바로 풀사이즈로 돌지 않는다는 점입니다.  
`reports/real-trade-readiness.json`의 gate를 통과하기 전에는 실계좌가 자동 보류됩니다.

## 프로젝트 구조

```text
.
├── README.md
├── run_bot.py
├── run_backtest.py
├── run_backtest_intraday.py
├── bot_ctl.sh
├── dev
├── docs/
│   ├── live-trading-checklist.md
│   ├── kakao-alert-setup.md
│   └── kakao-setup-checklist.md
├── scripts/
│   └── kakao_oauth_helper.py
├── src/
│   ├── account.py
│   ├── api_client.py
│   ├── auth.py
│   ├── config.py
│   ├── logger_setup.py
│   ├── main.py
│   ├── market_data.py
│   ├── notifications.py
│   ├── performance_reporting.py
│   ├── scheduler.py
│   ├── trading.py
│   ├── backtest/
│   │   ├── data_fetcher.py
│   │   ├── engine.py
│   │   ├── intraday_engine.py
│   │   └── report.py
│   └── strategies/
│       ├── momentum_scalp.py
│       └── regime_router.py
├── logs/
├── reports/
├── state/
├── data/
│   └── daily/
├── requirements.txt
└── open-trading-api/
```

## 실행 전 준비

### 1. 가상환경

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

프로젝트 기본 실행 진입점은 `./dev`입니다.

```bash
./dev py -V
./dev shell
```

### 2. 환경변수

`.env`에 최소 아래 값이 필요합니다.

```env
TRADING_MODE=paper

PAPER_API_KEY=...
PAPER_API_SECRET=...
PAPER_ACCOUNT_NUMBER=12345678

REAL_API_KEY=...
REAL_API_SECRET=...
REAL_ACCOUNT_NUMBER=12345678

ACCOUNT_PRODUCT_CODE=01
HTS_ID=...
LOG_LEVEL=INFO
OFF_HOURS_CHECK_INTERVAL_SECONDS=1800

ALLOW_DAILY_HARD_STOP_BYPASS=false

ALERTS_ENABLED=true
ALERT_CHANNEL=kakao
KAKAO_REST_API_KEY=...
KAKAO_CLIENT_SECRET=...
KAKAO_REDIRECT_URI=...
KAKAO_REFRESH_TOKEN=...
KAKAO_MESSAGE_WEB_URL=...
KAKAO_MESSAGE_MOBILE_WEB_URL=...
```

`TRADING_MODE=paper`이면 모의투자 계정 정보를, `TRADING_MODE=real`이면 실계좌 정보를 사용합니다.

카카오 알림을 쓰려면 `KAKAO_REFRESH_TOKEN`과 메시지 링크용 URL이 추가로 필요합니다.

## 카카오 알림 설정

카카오 알림은 `나에게 보내기` API를 사용합니다.

1회 토큰 발급용 URL 생성:

```bash
./dev py scripts/kakao_oauth_helper.py auth-url \
  --rest-api-key "$KAKAO_REST_API_KEY" \
  --redirect-uri "$KAKAO_REDIRECT_URI"
```

로그인 후 받은 `code`를 토큰으로 교환:

```bash
./dev py scripts/kakao_oauth_helper.py exchange-code \
  --rest-api-key "$KAKAO_REST_API_KEY" \
  --client-secret "$KAKAO_CLIENT_SECRET" \
  --redirect-uri "$KAKAO_REDIRECT_URI" \
  --code "인가코드"
```

반환된 `refresh_token`을 `.env`의 `KAKAO_REFRESH_TOKEN`에 넣고 봇을 재시작하면 됩니다.

## 자주 쓰는 명령

### 백테스트

최근 40일 기준 일봉 smoke 백테스트:

```bash
./dev py run_backtest.py
```

기간 지정:

```bash
./dev py run_backtest.py --days 40
./dev py run_backtest.py --days 60 --end-date 20260312
```

주의:

- 이 백테스트는 `장중 스캘프 전략을 완벽 재현하는 도구`가 아니라 `이상한 진입/리스크 구조를 빨리 걸러내는 smoke test`에 가깝습니다.
- `paper` 모드에서는 KIS 1분봉 API를 공식 검증 경로로 쓰지 못합니다.

### 1분봉 백테스트

```bash
./dev py run_backtest_intraday.py
```

주의:

- `paper` 모드에서는 이 스크립트가 바로 종료됩니다.
- 사실상 `real API 사용 가능 환경`에서만 실험용으로 쓸 수 있습니다.

### 유닛 테스트

```bash
./dev unit tests.test_risk_controls_unittest
./dev unit tests.test_backtest_engine_unittest
./dev unit tests.test_performance_reporting_unittest
```

또는 pytest:

```bash
./dev test -q
```

## 봇 실행과 운영

### 직접 실행

```bash
./dev py run_bot.py
```

### launchd 관리

실운영은 `bot_ctl.sh` 기준입니다.

```bash
./bot_ctl.sh install
./bot_ctl.sh start
./bot_ctl.sh stop
./bot_ctl.sh restart
./bot_ctl.sh status
./bot_ctl.sh today
./bot_ctl.sh report
./bot_ctl.sh gate
./bot_ctl.sh logs
./bot_ctl.sh monitor
./bot_ctl.sh uninstall
```

각 명령 의미:

- `status`: 실행 상태 + 오늘 손익 + 최근 로그
- `today`: 실행 상태 + 오늘 손익만 간단 출력
- `report`: 오늘 일일 성적표 출력
- `gate`: 실계좌 전환 게이트 출력
- `monitor`: 장중 핵심 이벤트만 필터링해서 보기

macOS에서는 `run_bot.py`가 `caffeinate -dims`를 띄워 절전을 막습니다.

## 리포트와 생성 파일

이 프로젝트는 장중 결과를 파일로 계속 남깁니다.

- `logs/trading.log`
전략 판단, 레짐 계산, 진입 거부, 체결, 리스크 이벤트 로그

- `logs/orders.log`
주문 성공/실패 로그

- `reports/YYYY/MM/daily-scorecard.YYYY-MM-DD.{json,md}`
일일 손익, 전략별 손익, 차단 사유, 최근 gate 상태

- `reports/real-trade-readiness.{json,md}`
paper 누적 성과 기준 실계좌 전환 가능 여부

- `state/momentum_scalp_daily_state.json`
당일 누적 손익/보유/중립장 손실 카운트 등 재시작 복구용 상태

- `data/daily/*.parquet`
일봉 백테스트 캐시

현재 `.gitignore`에는 아래 경로들이 이미 제외되어 있습니다.

- `logs/`
- `reports/`
- `state/momentum_scalp_daily_state.json`
- `data/daily/`

## 실계좌 전환 방식

실계좌는 바로 풀사이즈로 돌리지 않습니다.

1. `paper`에서 일일 성적표가 누적됩니다.
2. `reports/real-trade-readiness.json`에서 gate를 계산합니다.
3. gate 통과 전에는 `TRADING_MODE=real`이어도 자동 보류됩니다.
4. 통과 후에도 `25% -> 50% -> 100%` 단계로만 승격됩니다.

즉, 이 프로젝트의 기본 철학은 `paper에서 먼저 검증하고, 실계좌는 자동으로 보수적으로 제한`입니다.

## 운영상 알고 계셔야 할 점

- `neutral` 장세는 현재 가장 까다로운 구간입니다.
- 손실 직후에는 일부 후보가 `neutral_loss_cooldown`, `neutral_post_loss_quality_block`, `neutral_loss_limit_block`으로 거절될 수 있습니다.
- `neutral_loss_limit_block`으로 막힌 후보는 이제 그림자 추적으로 결과를 따로 남깁니다.
- 장마감 직전에는 신규 진입을 막도록 되어 있습니다.
- KIS API 제한 때문에 백테스트 데이터 다운로드 중 `초당 거래건수를 초과하였습니다`가 나올 수 있습니다.

## 문서

- [실투자 전환 체크리스트](docs/live-trading-checklist.md)
- [카카오 알림 설정](docs/kakao-alert-setup.md)
- [카카오 설정 체크리스트](docs/kakao-setup-checklist.md)

## 참고 저장소

`open-trading-api/`는 한국투자증권 샘플 레포를 참고용으로 클론해 둔 디렉터리입니다.  
실제 자동매매 로직은 이 프로젝트의 `src/` 아래에 있습니다.

## 면책

이 프로젝트는 개인 연구 및 자동화 실험용입니다.  
실제 투자 손실에 대한 책임은 사용자 본인에게 있습니다.
