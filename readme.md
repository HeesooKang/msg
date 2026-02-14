# 주식 자동거래 프로그램 (KIS Open API)

한국투자증권(Korea Investment & Securities) **Open API**를 사용하는 주식 자동거래 프로그램입니다.

## 개요

- **API**: [한국투자증권 Open API](https://apiportal.koreainvestment.com/) (KIS Developers)
- **전략**: 제가 설계한 전략을 **Claude Code**에게 구현 요청하여 작성했습니다.
- **기능**: 실시간 자동매매 봇, 백테스트 엔진, 모멘텀 스캘핑 전략 등

## 전략 요약

- **모멘텀 스캘핑**: 시가대비 상승, 등락률, 고가 근접도, 거래량 등 모멘텀 점수 기반 매수
- **시장 필터**: KOSPI MA20 등 레짐 필터 적용
- **매도**: 익절(+1.5%) / 개별 손절 / 추적손절 / 장마감 청산
- **일일 관리**: 목표 수익·최대 손실 도달 시 전량 청산 후 거래 중지
- **인버스 ETF**: 약세 구간에서 인버스 ETF 매매 (공매도 대체)

## 프로젝트 구조

```
.
├── run_bot.py          # 자동매매 봇 진입점 (스케줄러가 실행)
├── run_backtest.py     # 백테스트 실행
├── bot_ctl.sh          # 봇 스케줄러 관리 (launchd 설치/시작/중지 등)
├── docs/               # 운영/전환/알림 관련 문서
├── src/
│   ├── config.py       # 설정 로드
│   ├── auth.py         # KIS 토큰 관리
│   ├── api_client.py   # KIS API 클라이언트
│   ├── market_data.py  # 시세·호가 등 시장 데이터
│   ├── trading.py      # 주문·체결 처리
│   ├── strategies/
│   │   └── momentum_scalp.py  # 모멘텀 스캘핑 전략
│   └── backtest/       # 백테스트 엔진·리포트
├── open-trading-api/   # 한국투자증권 샘플 레포 (MCP용, 아래 참고)
└── requirements.txt
```

## `docs/` 폴더 안내

운영/전환/알림 설정 시 사람이 바로 참고할 체크리스트 문서를 모아둔 디렉터리입니다.

- `docs/live-trading-checklist.md`: 모의(`paper`) → 실전(`real`) 전환 체크리스트
- `docs/slack-alert-setup.md`: 슬랙 Incoming Webhook 알림 설정 가이드
- `docs/slack-setup-checklist.md`: 슬랙 설정 절차 요약 체크리스트

## `open-trading-api/` 폴더 안내

**`/open-trading-api/`** 폴더는 **MCP(Model Context Protocol)** 사용을 위해  
**한국투자증권 공식 GitHub 저장소**의 Open API 샘플 코드 저장소를 **클론해 둔 디렉터리**입니다.

- 원본: 한국투자증권에서 제공하는 KIS Open API 샘플 코드 저장소 (LLM 지원)
- 용도: MCP/LLM이 API 스펙·예제를 참고할 수 있도록 프로젝트 내에 포함
- 이 폴더의 코드는 참고용이며, 실제 자동거래 로직은 상위 `src/`에서 구현합니다.

## 설치 및 실행

### 요구 사항

- Python 3.10+
- 한국투자증권 계좌 및 Open API 앱 키·시크릿

### 설치

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 개발 명령 (venv 자동 고정)

`activate`를 매번 하지 않도록, 프로젝트에서는 `./dev`를 기본 실행 진입점으로 사용합니다.

```bash
./dev py run_backtest.py
./dev test -q --maxfail=1 tests
./dev unit tests/test_api_connection.py
./dev shell
```

### 설정

- `.env` 또는 설정 파일에 KIS API 앱 키, 시크릿, 계좌번호 등을 설정합니다.
- `src/config.py`에서 로드하는 방식을 확인하세요.
- 실전 전환 절차는 `docs/live-trading-checklist.md`를 참고하세요.
- 슬랙 알림 설정은 `docs/slack-alert-setup.md`를 참고하세요.

### 자동매매 봇 실행 (스케줄러)

이 봇은 **스케줄러(launchd)**에 의해 실행됩니다. 봇의 설치·시작·중지 등은 **`bot_ctl.sh`**로 관리합니다.

```bash
./bot_ctl.sh install   # launchd에 등록 (최초 1회, 로그인 시 자동 시작)
./bot_ctl.sh start     # 봇 시작
./bot_ctl.sh stop      # 봇 중지
./bot_ctl.sh restart   # 봇 재시작
./bot_ctl.sh status    # 상태 확인
./bot_ctl.sh today     # 오늘 손익 + 실행 상태 간단 확인
./bot_ctl.sh logs      # 로그 실시간 확인
./bot_ctl.sh uninstall # launchd에서 제거
```

- 스케줄러가 `run_bot.py`를 실행하며, 봇 내부에서는 10초 간격으로 전략이 동작합니다.
- macOS에서는 `caffeinate`으로 절전 방지를 시도합니다.

### 오늘 손익/실행 여부 빠르게 확인

```bash
./bot_ctl.sh today
```

- `실행 중/실행 안 됨` 상태와 함께, 오늘 기준 최신 `최종 잔고` 로그에서 손익을 보여줍니다.
- 아직 집계 로그가 없으면 `오늘 손익 로그가 아직 없습니다.`로 표시됩니다.

### 백테스트 실행

```bash
./dev py run_backtest.py
```

- 시총 상위 30종목 + 인버스 ETF 기준, 최근 약 2개월 데이터로 백테스트를 수행합니다.

## 라이선스 및 책임

- 본 프로그램은 개인 투자/연구 목적으로 작성되었습니다.
- `open-trading-api/` 내 샘플 코드에 대한 유의사항은 해당 폴더의 README를 참고하세요.
- 실제 투자 손실에 대한 책임은 사용자에게 있습니다.
