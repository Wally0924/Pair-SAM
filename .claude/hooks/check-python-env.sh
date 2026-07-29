#!/usr/bin/env bash
# PreToolUse(Bash) hook — PairSAM 規則:Python/pip 必須走 sam_env。
# 例外:conda run -n sam_env、sam_env 的絕對路徑、python -c 快速檢查、py_compile。
# 另外:偵測到 pytest 執行時,清除「已改 .py 未測試」的 flag(供 Stop hook 使用)。
input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // empty')
sid=$(printf '%s' "$input" | jq -r '.session_id // empty')
[ -z "$cmd" ] && exit 0

if printf '%s' "$cmd" | grep -q 'pytest'; then
  [ -n "$sid" ] && rm -f "/tmp/claude-wsam-pyedit-${sid}"
fi

case "$cmd" in
  *"conda run -n sam_env"*|*"envs/sam_env/bin/"*|*"conda activate sam_env"*|*py_compile*) exit 0 ;;
esac

# python/pip -c 快速檢查不需要 sam_env
if printf '%s' "$cmd" | grep -qE '(^|[;&|][[:space:]]*)(python3?)[[:space:]]+-c([[:space:]]|$)'; then
  exit 0
fi

if printf '%s' "$cmd" | grep -qE '(^|[;&|][[:space:]]*)(python3?|pip3?)([[:space:]]|$)'; then
  cat <<'EOF'
{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"PairSAM 規則:Python/pip 必須在 sam_env 中執行。請改用 conda run -n sam_env python ... 或 sam_env 的絕對路徑 <conda-root>/envs/sam_env/bin/python(python -c 快速檢查與 py_compile 例外)。"}}
EOF
fi
exit 0
