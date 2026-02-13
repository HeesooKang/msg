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

PLIST_NAME="com.kis.trading-bot"
PLIST_SRC="$HOME/msg/com.kis.trading-bot.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"
TRADING_LOG="$HOME/msg/logs/trading.log"

print_service_status() {
    echo "=== 서비스 상태 ==="
    launchctl list | grep "$PLIST_NAME" >/dev/null && echo "→ 실행 중" || echo "→ 실행 안 됨"
}

print_today_pnl() {
    local today_line pnl balance ts
    today_line=$(grep "^$(date +%F) .*최종 잔고" "$TRADING_LOG" 2>/dev/null | tail -1)

    echo "=== 오늘 손익 ==="
    if [ -n "$today_line" ]; then
        ts=$(echo "$today_line" | awk '{print $1" "$2}')
        balance=$(echo "$today_line" | awk -F'평가금액: | \\| 손익: ' '{print $2}')
        pnl=$(echo "$today_line" | awk -F'손익: ' '{print $2}')
        echo "→ 최신 집계 시각: $ts"
        echo "→ 평가금액: $balance"
        echo "→ 손익: $pnl"
    else
        echo "→ 오늘 손익 로그가 아직 없습니다."
    fi
}

case "$1" in
    install)
        cp "$PLIST_SRC" "$PLIST_DST"
        echo "✓ plist 설치 완료: $PLIST_DST"
        echo "  → 다음 로그인 시 자동 시작됩니다."
        echo "  → 지금 바로 시작하려면: ./bot_ctl.sh start"
        ;;
    start)
        launchctl load "$PLIST_DST" 2>/dev/null
        launchctl start "$PLIST_NAME"
        echo "✓ 봇 시작됨"
        ;;
    stop)
        launchctl stop "$PLIST_NAME"
        echo "✓ 봇 중지됨"
        ;;
    restart)
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
    *)
        echo "사용법: $0 {install|start|stop|restart|status|today|uninstall|logs}"
        exit 1
        ;;
esac
