#!/bin/bash
# KIS 자동매매 봇 관리 스크립트
# 사용법:
#   ./bot_ctl.sh install   — launchd에 등록 (최초 1회)
#   ./bot_ctl.sh start     — 봇 시작
#   ./bot_ctl.sh stop      — 봇 중지
#   ./bot_ctl.sh restart   — 봇 재시작
#   ./bot_ctl.sh status    — 상태 확인
#   ./bot_ctl.sh today     — 오늘 손익 + 실행 상태 간단 확인
#   ./bot_ctl.sh uninstall — launchd에서 제거
#   ./bot_ctl.sh logs      — 로그 실시간 확인
#   ./bot_ctl.sh monitor   — 장중 핵심 이벤트 필터 로그

PLIST_NAME="com.kis.trading-bot"
PLIST_SRC="$HOME/msg/com.kis.trading-bot.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"
TRADING_LOG="$HOME/msg/logs/trading.log"

print_service_status() {
    echo "=== 서비스 상태 ==="
    launchctl list | grep "$PLIST_NAME" >/dev/null && echo "→ 실행 중" || echo "→ 실행 안 됨"
}

print_today_pnl() {
    local today final_line realized_line legacy_line latest_line pnl balance ts
    today=$(date +%F)
    final_line=$(grep -h "^$today .*최종 잔고" "$TRADING_LOG" "$TRADING_LOG.$today" 2>/dev/null | tail -1)

    echo "=== 오늘 손익 ==="
    if [ -n "$final_line" ]; then
        ts=$(echo "$final_line" | awk '{print $1" "$2}')
        balance=$(echo "$final_line" | awk -F'평가금액: | \\| 손익: ' '{print $2}')
        pnl=$(echo "$final_line" | awk -F'손익: ' '{print $2}')
        echo "→ 최신 집계 시각: $ts"
        echo "→ 기준: 세션 종료 잔고 요약"
        echo "→ 평가금액: $balance"
        echo "→ 손익: $pnl"
        return
    fi

    realized_line=$(grep -h "^$today .*누적순손익:" "$TRADING_LOG" "$TRADING_LOG.$today" 2>/dev/null | tail -1)
    if [ -n "$realized_line" ]; then
        ts=$(echo "$realized_line" | awk '{print $1" "$2}')
        pnl=$(echo "$realized_line" | sed -nE 's/.*누적순손익: ([^)]*).*/\1/p')
        echo "→ 최신 집계 시각: $ts"
        echo "→ 기준: 최근 체결 누적순손익"
        echo "→ 손익: $pnl"
        return
    fi

    legacy_line=$(grep -h "^$today .*누적: " "$TRADING_LOG" "$TRADING_LOG.$today" 2>/dev/null | tail -1)
    if [ -n "$legacy_line" ]; then
        ts=$(echo "$legacy_line" | awk '{print $1" "$2}')
        pnl=$(echo "$legacy_line" | sed -nE 's/.*누적: ([^)]*).*/\1/p')
        echo "→ 최신 집계 시각: $ts"
        echo "→ 기준: 최근 체결 누적손익"
        echo "→ 손익: $pnl"
        return
    fi

    latest_line=$(grep -h "^$today " "$TRADING_LOG" "$TRADING_LOG.$today" 2>/dev/null | tail -1)
    if [ -n "$latest_line" ]; then
        ts=$(echo "$latest_line" | awk '{print $1" "$2}')
        echo "→ 최신 집계 시각: $ts"
        echo "→ 기준: 오늘 실현손익 로그 미발생"
        echo "→ 손익: 0원"
        return
    fi

    echo "→ 오늘 손익 로그가 아직 없습니다."
}

monitor_filtered_logs() {
    local pattern
    pattern='전략 초기화|오늘 적용값 요약|신규 진입 차단 시간대|눌림목 필터|눌림목 대기|눌림목 이탈|눌림목 취소|매수 신호|추가매수 신호|매수 체결|매도 체결|익절|개별손절|추적손절|전역 리스크 쿨다운|시장반등 청산|시간초과 청산|일일 손실한도 도달|보조 손실컷 도달|장마감 임박|최종 잔고'

    if command -v rg >/dev/null 2>&1; then
        tail -f "$TRADING_LOG" 2>/dev/null | rg --line-buffered -e "$pattern"
    else
        tail -f "$TRADING_LOG" 2>/dev/null | grep --line-buffered -E "$pattern"
    fi
}

load_service() {
    if [ ! -f "$PLIST_DST" ]; then
        echo "❌ 실행 파일이 없습니다: $PLIST_DST"
        return 1
    fi
    launchctl load -w "$PLIST_DST" >/dev/null 2>&1
}

case "$1" in
    install)
        cp "$PLIST_SRC" "$PLIST_DST"
        echo "✓ plist 설치 완료: $PLIST_DST"
        echo "  → 다음 로그인 시 자동 시작됩니다."
        echo "  → 지금 바로 시작하려면: ./bot_ctl.sh start"
        ;;
    start)
        load_service
        launchctl load "$PLIST_DST" 2>/dev/null
        launchctl start "$PLIST_NAME"
        echo "✓ 봇 시작됨"
        ;;
    stop)
        launchctl stop "$PLIST_NAME"
        echo "✓ 봇 중지됨"
        ;;
    restart)
        load_service
        launchctl stop "$PLIST_NAME" 2>/dev/null
        sleep 2
        launchctl start "$PLIST_NAME"
        echo "✓ 봇 재시작됨"
        ;;
    status)
        print_service_status
        echo ""
        print_today_pnl
        echo ""
        echo "=== 최근 로그 (5줄) ==="
        tail -5 "$TRADING_LOG" 2>/dev/null || echo "(로그 없음)"
        ;;
    today)
        print_service_status
        echo ""
        print_today_pnl
        ;;
    uninstall)
        launchctl stop "$PLIST_NAME" 2>/dev/null
        launchctl unload "$PLIST_DST" 2>/dev/null
        rm -f "$PLIST_DST"
        echo "✓ plist 제거 완료"
        ;;
    logs)
        tail -f "$TRADING_LOG"
        ;;
    monitor)
        monitor_filtered_logs
        ;;
    *)
        echo "사용법: $0 {install|start|stop|restart|status|today|uninstall|logs|monitor}"
        exit 1
        ;;
esac
