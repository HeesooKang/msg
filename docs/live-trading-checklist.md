# 실전 전환 체크리스트

이 문서는 모의(`paper`)에서 실전(`real`)으로 전환할 때의 최소 점검 절차를 정리합니다.

## 1) 전환 전 점검

- 최근 모의 운용 로그에서 아래 이벤트가 정상인지 확인
  - `순실현` 기준 일일 하드스탑
  - `보조 손실컷 도달`
  - `매도 실패(포지션 유지)`
- 계좌상품코드 확인 (`ACCOUNT_PRODUCT_CODE`)
  - 보통 주식 종합계좌는 `01`

## 2) `.env` 변경

- `TRADING_MODE=real`
- 실전 키/계좌 입력
  - `REAL_API_KEY`
  - `REAL_API_SECRET`
  - `REAL_ACCOUNT_NUMBER`
- `HTS_ID` 값 확인

주의:
- 모의로 되돌릴 때는 `TRADING_MODE=paper`로 변경
- 토큰 파일은 모드별로 분리 저장됨 (`KIS_paper_YYYYMMDD`, `KIS_real_YYYYMMDD`)

## 3) 재시작

```bash
./bot_ctl.sh restart
```

## 4) 전환 직후 즉시 확인

```bash
./bot_ctl.sh status
./bot_ctl.sh logs
```

- 시작 로그에 모드가 `[REAL]`인지 확인
- API 인증/주문 관련 에러가 없는지 확인

## 5) 장중 모니터링 포인트

- 주문 성공 후 체결가 보정 로그/손익 반영이 자연스러운지 확인
- 비정상 반복 주문/취소가 없는지 확인
- 하드스탑 발생 시 전량 청산 후 신규 진입 중단되는지 확인

## 6) 롤백 절차 (실전 -> 모의)

1. `.env`에서 `TRADING_MODE=paper`로 변경
2. `PAPER_*` 키/계좌값 확인
3. `./bot_ctl.sh restart`
4. 로그에서 `[PAPER]` 확인
