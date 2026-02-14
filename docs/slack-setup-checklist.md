# 슬랙 알림 설정 체크리스트

## 네가 슬랙에서 해야 할 일

1. 슬랙 워크스페이스 준비(없으면 생성)
2. Incoming Webhooks 앱 추가
3. Add New Webhook to Workspace로 알림 채널 선택
4. Webhook URL 복사

## 프로젝트에서 네가 해야 할 일

1. `.env`에 아래 추가

```env
ALERTS_ENABLED=true
ALERT_CHANNEL=slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
ALERT_MIN_INTERVAL_SECONDS=300
```

2. 봇 재시작

```bash
./bot_ctl.sh restart
```

3. 로그/슬랙 수신 확인

```bash
./bot_ctl.sh logs
```
