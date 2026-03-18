# 카카오 알림 설정 가이드

이 프로젝트는 카카오톡 메시지 API의 `나에게 보내기`를 사용해 운영 이벤트 알림을 보낼 수 있습니다.

## 1) 카카오에서 할 일

1. 카카오 디벨로퍼스 앱 생성
- `https://developers.kakao.com/console/app` 에 접속해 앱을 만듭니다.

2. 카카오 로그인 활성화
- 앱 설정에서 `카카오 로그인`을 활성화합니다.

3. Redirect URI 등록
- `카카오 로그인 > Redirect URI`에 토큰 발급용 URI를 등록합니다.
- 예: `http://localhost:9876/kakao/callback`

4. Client Secret 확인
- `카카오 로그인 > 보안`에서 Client Secret을 확인합니다.
- 카카오 공식 문서상 REST API 키의 Client Secret 기능은 기본 활성화 상태일 수 있습니다.

5. 동의항목 설정
- `카카오 로그인 > 동의항목`에서 `카카오톡 메시지 전송(talk_message)`를 사용하도록 설정합니다.

6. 제품 링크 관리
- `앱 > 제품 링크 관리`에서 메시지에 넣을 웹 도메인을 등록합니다.
- 이 도메인이 `.env`의 `KAKAO_MESSAGE_WEB_URL`과 같아야 합니다.

## 2) 프로젝트에서 할 일

`.env`에 아래 값 추가:

```env
ALERTS_ENABLED=true
ALERT_CHANNEL=kakao
KAKAO_REST_API_KEY=카카오_앱_REST_API_키
KAKAO_CLIENT_SECRET=카카오_클라이언트_시크릿
KAKAO_REDIRECT_URI=http://localhost:9876/kakao/callback
KAKAO_REFRESH_TOKEN=1회_로그인으로_받은_refresh_token
KAKAO_MESSAGE_WEB_URL=https://example.com
KAKAO_MESSAGE_MOBILE_WEB_URL=https://example.com
KAKAO_MESSAGE_BUTTON_TITLE=상세 보기
ALERT_MIN_INTERVAL_SECONDS=300
```

## 3) 1회 토큰 발급

인가 URL 생성:

```bash
./dev py scripts/kakao_oauth_helper.py auth-url \
  --rest-api-key "$KAKAO_REST_API_KEY" \
  --redirect-uri "$KAKAO_REDIRECT_URI"
```

브라우저에서 로그인/동의 후 `redirect_uri?code=...` 형태로 이동하면, URL의 `code` 값을 복사합니다.

토큰 교환:

```bash
./dev py scripts/kakao_oauth_helper.py exchange-code \
  --rest-api-key "$KAKAO_REST_API_KEY" \
  --client-secret "$KAKAO_CLIENT_SECRET" \
  --redirect-uri "$KAKAO_REDIRECT_URI" \
  --code "복사한_인가코드"
```

출력된 `refresh_token`을 `.env`의 `KAKAO_REFRESH_TOKEN`에 저장합니다.

## 4) 봇 재시작

```bash
./bot_ctl.sh restart
```

## 5) 알림 대상 이벤트

- 일일 손실한도 도달
- 일일 목표 달성
- 보조 손실컷 도달
- 장마감 전량 청산
- 매도 실패

## 6) 문제 발생 시 점검

- `KAKAO_REFRESH_TOKEN`이 만료되거나 잘못되지 않았는지 확인
- `KAKAO_MESSAGE_WEB_URL` 도메인이 카카오 제품 링크 관리에 등록되어 있는지 확인
- `logs/trading.log`에서 `kis_trader.notifications` 경고 로그 확인
- 카카오 메시지 API는 과금 정책이 있을 수 있으니 공식 쿼터/요금 문서를 확인
