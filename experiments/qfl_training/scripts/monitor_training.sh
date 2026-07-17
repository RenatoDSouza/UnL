#!/usr/bin/env bash
set -u

ROOT="/home/rdias/UnL"
SESSION="qfl_training"
TRAINING_LOG="$ROOT/experiments/qfl_training/logs/training_20260716_1208.log"
MONITOR_LOG="$ROOT/experiments/qfl_training/logs/training_monitor.log"

check_training() {
    local timestamp last_lines errors
    timestamp="$(date -Is)"
    last_lines="$(tail -n 40 "$TRAINING_LOG" 2>&1)"
    errors="$(printf '%s\n' "$last_lines" | grep -Ei 'traceback|exception|error|failed|killed|segmentation fault' || true)"

    {
        printf '\n[%s] ' "$timestamp"
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            printf 'STATUS=running\n'
        else
            printf 'STATUS=not_running\n'
        fi
        if [[ -n "$errors" ]]; then
            printf 'ALERT: error signature found in recent training log:\n%s\n' "$errors"
        else
            printf 'No error signature found in the recent training log.\n'
        fi
        printf '%s\n' "$last_lines"
    } >> "$MONITOR_LOG"
}

# Six checks at five-minute intervals cover the first 30 minutes.
for _ in {1..6}; do
    check_training
    sleep 300
done

# Continue checking hourly until the training tmux session ends.
while tmux has-session -t "$SESSION" 2>/dev/null; do
    check_training
    sleep 3600
done

check_training
