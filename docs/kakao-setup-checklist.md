# 카카오 알림 설정 체크리스트

## 카카오 디벨로퍼스에서 하실 일

1. 앱 생성
2. `카카오 로그인` 활성화
3. `Redirect URI` 등록
4. `Client Secret` 확인
5. `카카오톡 메시지 전송(talk_message)` 동의항목 설정
6. `제품 링크 관리`에 메시지용 웹 도메인 등록

## 프로젝트에서 하실 일

1. `.env`에 아래 추가

```env
ALERTS_ENABLED=true
ALERT_CHANNEL=kakao
KAKAO_REST_API_KEY=...
KAKAO_CLIENT_SECRET=...
KAKAO_REDIRECT_URI=...
KAKAO_REFRESH_TOKEN=...
KAKAO_MESSAGE_WEB_URL=...
KAKAO_MESSAGE_MOBILE_WEB_URL=...
ALERT_MIN_INTERVAL_SECONDS=300
```

2. 인가 URL 생성

```bash
./dev py scripts/kakao_oauth_helper.py auth-url \
  --rest-api-key "$KAKAO_REST_API_KEY" \
  --redirect-uri "$KAKAO_REDIRECT_URI"
```

3. 인가 코드 교환 후 `KAKAO_REFRESH_TOKEN` 저장

4. 봇 재시작

```bash
./bot_ctl.sh restart
```

5. 로그/카카오톡 수신 확인

```bash
./bot_ctl.sh logs
```
