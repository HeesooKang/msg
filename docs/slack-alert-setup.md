# 슬랙 알림 설정 가이드

이 프로젝트는 슬랙 Incoming Webhook으로 운영 이벤트 알림을 보낼 수 있습니다.

## 1) 슬랙에서 할 일

1. 슬랙 워크스페이스 준비
- 워크스페이스가 없으면 새로 생성합니다.

2. Incoming Webhooks 앱 추가
- 슬랙 앱 디렉터리에서 `Incoming Webhooks`를 검색해 워크스페이스에 추가합니다.

3. Webhook URL 발급
- `Add New Webhook to Workspace` 클릭
- 알림 받을 채널을 선택
- 발급된 Webhook URL 복사

## 2) 프로젝트에서 할 일

`.env`에 아래 값 추가:

```env
ALERTS_ENABLED=true
ALERT_CHANNEL=slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
ALERT_MIN_INTERVAL_SECONDS=300
```

- `ALERT_MIN_INTERVAL_SECONDS`: 동일 이벤트 중복 전송 최소 간격(초)

## 3) 봇 재시작

```bash
./bot_ctl.sh restart
```

## 4) 알림 대상 이벤트

- 일일 손실한도 도달
- 일일 목표 달성
- 보조 손실컷 도달
- 장마감 전량 청산
- 매도 실패(포지션 유지)

## 5) 문제 발생 시 점검

- Webhook URL이 정확한지 확인
- 슬랙 채널 권한/보관정책으로 메시지가 막히지 않는지 확인
- `logs/trading.log`에서 `kis_trader.notifications` 경고 로그 확인
