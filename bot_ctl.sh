#!/bin/bash
# KIS 자동매매 봇 관리 스크립트
# 사용법:
#   ./bot_ctl.sh install   — launchd에 등록 (최초 1회)
#   ./bot_ctl.sh start     — 봇 시작
#   ./bot_ctl.sh stop      — 봇 중지
#   ./bot_ctl.sh restart   — 봇 재시작
#   ./bot_ctl.sh status    — 상태 확인
#   ./bot_ctl.sh uninstall — launchd에서 제거
#   ./bot_ctl.sh logs      — 로그 실시간 확인

PLIST_NAME="com.kis.trading-bot"
PLIST_SRC="$HOME/msg/com.kis.trading-bot.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"

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
        echo "=== 서비스 상태 ==="
        launchctl list | grep "$PLIST_NAME" && echo "→ 실행 중" || echo "→ 실행 안 됨"
        echo ""
        echo "=== 최근 로그 (5줄) ==="
        tail -5 "$HOME/msg/logs/trading.log" 2>/dev/null || echo "(로그 없음)"
        ;;
    uninstall)
        launchctl stop "$PLIST_NAME" 2>/dev/null
        launchctl unload "$PLIST_DST" 2>/dev/null
        rm -f "$PLIST_DST"
        echo "✓ plist 제거 완료"
        ;;
    logs)
        tail -f "$HOME/msg/logs/trading.log"
        ;;
    *)
        echo "사용법: $0 {install|start|stop|restart|status|uninstall|logs}"
        exit 1
        ;;
esac
