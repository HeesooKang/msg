#!/bin/bash
# KIS 자동매매 봇 관리 스크립트
# 사용법:
#   ./bot_ctl.sh install   — launchd에 등록 (최초 1회)
#   ./bot_ctl.sh start     — 봇 시작
#   ./bot_ctl.sh stop      — 봇 중지
#   ./bot_ctl.sh restart   — 봇 재시작
#   ./bot_ctl.sh status    — 상태 확인
#   ./bot_ctl.sh today     — 오늘 손익 + 실행 상태 간단 확인
#   ./bot_ctl.sh report    — 오늘 성적표 확인
#   ./bot_ctl.sh gate      — 실투자 전환 게이트 확인
#   ./bot_ctl.sh uninstall — launchd에서 제거
#   ./bot_ctl.sh logs      — 로그 실시간 확인
#   ./bot_ctl.sh monitor   — 장중 핵심 이벤트 필터 로그

PROJECT_ROOT="$HOME/msg"
PLIST_NAME="com.kis.trading-bot"
PLIST_SRC="$PROJECT_ROOT/com.kis.trading-bot.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"
TRADING_LOG="$PROJECT_ROOT/logs/trading.log"
REPORT_ROOT="$PROJECT_ROOT/reports"
LAUNCH_DOMAIN="gui/$(id -u)"
SERVICE_TARGET="$LAUNCH_DOMAIN/$PLIST_NAME"

current_ts() {
    date '+%F %T'
}

service_pid() {
    launchctl print "$SERVICE_TARGET" 2>/dev/null | awk '/pid = / {print $3; exit}'
}

service_start_time() {
    local pid
    pid=$(service_pid)
    if [ -z "$pid" ] || [ "$pid" -le 0 ] 2>/dev/null; then
        return 1
    fi
    ps -o lstart= -p "$pid" 2>/dev/null | sed 's/^ *//'
}

service_loaded() {
    launchctl print "$SERVICE_TARGET" >/dev/null 2>&1
}

launchctl_command_ok() {
    local output="$1" status="${2:-1}"
    if [ "$status" -ne 0 ]; then
        return 1
    fi
    if echo "$output" | grep -Eqi 'failed:|not privileged|operation not permitted|input/output error'; then
        return 1
    fi
    return 0
}

wait_for_service_start() {
    local timeout="${1:-15}" expected_new_pid="${2:-}" elapsed=0 pid
    while [ "$elapsed" -lt "$timeout" ]; do
        pid=$(service_pid)
        if [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null; then
            if [ -n "$expected_new_pid" ] && [ "$pid" = "$expected_new_pid" ]; then
                sleep 1
                elapsed=$((elapsed + 1))
                continue
            fi
            echo "$pid"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

wait_for_service_stop() {
    local timeout="${1:-15}" elapsed=0 pid
    while [ "$elapsed" -lt "$timeout" ]; do
        pid=$(service_pid)
        if [ -z "$pid" ] || [ "$pid" -le 0 ] 2>/dev/null; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

latest_log_timestamp() {
    local today latest_line
    today=$(date +%F)
    latest_line=$(grep -h "^$today " "$TRADING_LOG" "$TRADING_LOG.$today" 2>/dev/null | tail -1)
    if [ -z "$latest_line" ]; then
        return 1
    fi
    echo "$latest_line" | awk '{print $1" "$2}' | cut -d, -f1
}

latest_session_start_timestamp() {
    local today latest_line
    today=$(date +%F)
    latest_line=$(grep -h "^$today .*--- 트레이딩 세션 시작 ---" "$TRADING_LOG" "$TRADING_LOG.$today" 2>/dev/null | tail -1)
    if [ -z "$latest_line" ]; then
        return 1
    fi
    echo "$latest_line" | awk '{print $1" "$2}' | cut -d, -f1
}

wait_for_fresh_log_after() {
    local timeout="${1:-20}" reference_ts="${2:-}" elapsed=0 log_ts
    while [ "$elapsed" -lt "$timeout" ]; do
        log_ts=$(latest_log_timestamp)
        if [ -n "$log_ts" ] && [ -n "$reference_ts" ] && [[ "$log_ts" > "$reference_ts" || "$log_ts" = "$reference_ts" ]]; then
            echo "$log_ts"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

wait_for_session_start_after() {
    local timeout="${1:-20}" reference_ts="${2:-}" elapsed=0 session_ts
    while [ "$elapsed" -lt "$timeout" ]; do
        session_ts=$(latest_session_start_timestamp)
        if [ -n "$session_ts" ] && [ -n "$reference_ts" ] && [[ "$session_ts" > "$reference_ts" || "$session_ts" = "$reference_ts" ]]; then
            echo "$session_ts"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

print_service_status() {
    echo "=== 서비스 상태 ==="
    local pid
    pid=$(service_pid)
    if [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null; then
        echo "→ 실행 중 (pid: $pid)"
    else
        echo "→ 실행 안 됨"
    fi
}

print_runtime_alignment() {
    local start_ts session_ts
    start_ts=$(service_start_time)
    session_ts=$(latest_session_start_timestamp)
    echo "=== 런타임 기준 ==="
    if [ -n "$start_ts" ]; then
        echo "→ 현재 pid 시작 시각: $start_ts"
    else
        echo "→ 현재 pid 시작 시각: 확인 불가"
    fi
    if [ -n "$session_ts" ]; then
        echo "→ 최신 세션 시작 로그: $session_ts"
    else
        echo "→ 최신 세션 시작 로그: 없음"
    fi
}

print_today_pnl() {
    local today archive_log final_line realized_line latest_line pnl balance ts line_ts latest_ts
    today=$(date +%F)
    archive_log="$HOME/msg/logs/$(date +%Y)/$(date +%m)/trading.log.$today"
    final_line=$(grep -h "^$today .*최종 잔고" "$TRADING_LOG" "$TRADING_LOG.$today" "$archive_log" 2>/dev/null | tail -1)
    realized_line=$(grep -h "^$today .*매도 .* 확정: .*일실현=" "$TRADING_LOG" "$TRADING_LOG.$today" "$archive_log" 2>/dev/null | tail -1)

    for line in "$final_line" "$realized_line"; do
        [ -z "$line" ] && continue
        line_ts=$(echo "$line" | awk '{print $1" "$2}')
        if [ -z "$latest_line" ] || [[ "$line_ts" > "$latest_ts" ]]; then
            latest_line="$line"
            latest_ts="$line_ts"
        fi
    done

    echo "=== 오늘 손익 ==="
    if [ -n "$latest_line" ] && echo "$latest_line" | grep -q "최종 잔고"; then
        ts=$(echo "$latest_line" | awk '{print $1" "$2}')
        balance=$(echo "$latest_line" | awk -F'평가금액: | \\| 금일 실현손익: ' '{print $2}')
        pnl=$(echo "$latest_line" | awk -F'금일 실현손익: ' '{print $2}')
        echo "→ 최신 집계 시각: $ts"
        echo "→ 기준: 세션 종료 잔고 요약"
        echo "→ 평가금액: $balance"
        echo "→ 손익: $pnl"
        return
    fi

    if [ -n "$latest_line" ] && echo "$latest_line" | grep -q "일실현="; then
        ts=$(echo "$latest_line" | awk '{print $1" "$2}')
        pnl=$(echo "$latest_line" | sed -nE 's/.*일실현=([+-]?[0-9,]+원).*/\1/p')
        echo "→ 최신 집계 시각: $ts"
        echo "→ 기준: 최근 확정 매도체결 누적순손익"
        echo "→ 손익: $pnl"
        return
    fi

    latest_line=$(grep -h "^$today " "$TRADING_LOG" "$TRADING_LOG.$today" "$archive_log" 2>/dev/null | tail -1)
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
    pattern='EV 전략 초기화|동적풀 갱신|틱 상태|EV 배치|EV 매수 선택|신규 진입 종료|매수 체결|매도 체결|계획 청산|당일 손실 하드스탑|당일 목표 달성|최종 잔고|트레이딩 세션'

    if command -v rg >/dev/null 2>&1; then
        tail -f "$TRADING_LOG" 2>/dev/null | rg --line-buffered -e "$pattern"
    else
        tail -f "$TRADING_LOG" 2>/dev/null | grep --line-buffered -E "$pattern"
    fi
}

print_today_report() {
    PYTHONPATH="$PROJECT_ROOT" python3 -m src.performance_reporting today --report-root "$REPORT_ROOT"
}

print_real_trade_gate() {
    PYTHONPATH="$PROJECT_ROOT" python3 -m src.performance_reporting gate --report-root "$REPORT_ROOT"
}

load_service() {
    local output status
    if [ ! -f "$PLIST_DST" ]; then
        echo "❌ 실행 파일이 없습니다: $PLIST_DST"
        return 1
    fi
    if service_loaded; then
        return 0
    fi
    output=$(launchctl bootstrap "$LAUNCH_DOMAIN" "$PLIST_DST" 2>&1)
    status=$?
    if launchctl_command_ok "$output" "$status"; then
        return 0
    fi
    output=$(launchctl load -w "$PLIST_DST" 2>&1)
    status=$?
    if launchctl_command_ok "$output" "$status"; then
        return 0
    fi
    [ -n "$output" ] && echo "$output"
    return 1
}

unload_service() {
    local output status
    if ! service_loaded; then
        return 0
    fi
    output=$(launchctl bootout "$SERVICE_TARGET" 2>&1)
    status=$?
    if launchctl_command_ok "$output" "$status"; then
        return 0
    fi
    output=$(launchctl bootout "$LAUNCH_DOMAIN" "$PLIST_DST" 2>&1)
    status=$?
    if launchctl_command_ok "$output" "$status"; then
        return 0
    fi
    output=$(launchctl unload -w "$PLIST_DST" 2>&1)
    status=$?
    if launchctl_command_ok "$output" "$status"; then
        return 0
    fi
    [ -n "$output" ] && echo "$output"
    return 1
}

case "$1" in
    install)
        cp "$PLIST_SRC" "$PLIST_DST"
        echo "✓ plist 설치 완료: $PLIST_DST"
        echo "  → 다음 로그인 시 자동 시작됩니다."
        echo "  → 지금 바로 시작하려면: ./bot_ctl.sh start"
        ;;
    start)
        request_ts=$(current_ts)
        if current_pid=$(service_pid); then
            if [ -n "$current_pid" ] && [ "$current_pid" -gt 0 ] 2>/dev/null; then
                echo "✓ 봇이 이미 실행 중입니다 (pid: $current_pid)"
                exit 0
            fi
        fi
        if ! load_service; then
            echo "❌ 봇 시작 실패: launchd 서비스 로드에 실패했습니다."
            exit 1
        fi
        if pid=$(wait_for_service_start 20); then
            echo "✓ 봇 시작됨 (pid: $pid)"
            if log_ts=$(wait_for_fresh_log_after 20 "$request_ts"); then
                echo "→ 새 로그 확인: $log_ts"
            else
                echo "⚠️  시작 후 새 로그를 확인하지 못했습니다."
            fi
        else
            echo "❌ 봇 시작 실패: 프로세스가 올라오지 않았습니다."
            exit 1
        fi
        ;;
    stop)
        if ! unload_service; then
            echo "❌ 봇 중지 실패: launchd 서비스 언로드에 실패했습니다."
            exit 1
        fi
        if wait_for_service_stop 20; then
            echo "✓ 봇 중지됨"
        else
            echo "❌ 봇 중지 실패: 프로세스 종료를 확인하지 못했습니다."
            exit 1
        fi
        ;;
    restart)
        request_ts=$(current_ts)
        old_pid=$(service_pid)
        if [ -n "$old_pid" ] && [ "$old_pid" -gt 0 ] 2>/dev/null; then
            if kill -TERM "$old_pid" >/dev/null 2>&1; then
                if pid=$(wait_for_service_start 20 "$old_pid"); then
                    echo "✓ 봇 재시작됨 (pid: $pid)"
                    if session_ts=$(wait_for_session_start_after 20 "$request_ts"); then
                        echo "→ 새 세션 시작 로그 확인: $session_ts"
                    elif log_ts=$(wait_for_fresh_log_after 20 "$request_ts"); then
                        echo "→ 새 로그 확인: $log_ts"
                    else
                        echo "⚠️  재시작 후 새 로그를 확인하지 못했습니다."
                    fi
                    exit 0
                fi
            fi
        fi
        if ! unload_service; then
            echo "❌ 봇 재시작 실패: launchd 서비스 언로드에 실패했습니다."
            exit 1
        fi
        if ! wait_for_service_stop 20; then
            echo "❌ 봇 재시작 실패: 기존 프로세스 종료를 확인하지 못했습니다."
            exit 1
        fi
        if ! load_service; then
            echo "❌ 봇 재시작 실패: launchd 서비스 로드에 실패했습니다."
            exit 1
        fi
        if pid=$(wait_for_service_start 20 "$old_pid"); then
            echo "✓ 봇 재시작됨 (pid: $pid)"
            if session_ts=$(wait_for_session_start_after 20 "$request_ts"); then
                echo "→ 새 세션 시작 로그 확인: $session_ts"
            elif log_ts=$(wait_for_fresh_log_after 20 "$request_ts"); then
                echo "→ 새 로그 확인: $log_ts"
            else
                echo "⚠️  재시작 후 새 로그를 확인하지 못했습니다."
            fi
        else
            echo "❌ 봇 재시작 실패: 새 프로세스 시작을 확인하지 못했습니다."
            if [ -n "$old_pid" ] && [ "$(service_pid)" = "$old_pid" ]; then
                echo "→ 기존 pid($old_pid)가 유지되어 실제 재시작이 일어나지 않았습니다."
            fi
            exit 1
        fi
        ;;
    status)
        print_service_status
        echo ""
        print_runtime_alignment
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
    report)
        print_today_report
        ;;
    gate)
        print_real_trade_gate
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
        echo "사용법: $0 {install|start|stop|restart|status|today|report|gate|uninstall|logs|monitor}"
        exit 1
        ;;
esac
