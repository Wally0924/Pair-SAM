#!/usr/bin/env bash
# PostToolUse(Write|Edit) hook — .py 檔存檔後立即做語法檢查,失敗時把錯誤回饋給模型。
# 若改動的是 segment-anything 底下的 .py,另外記 flag 供 Stop hook 提醒跑測試。
input=$(cat)
f=$(printf '%s' "$input" | jq -r '.tool_input.file_path // .tool_response.filePath // empty')
sid=$(printf '%s' "$input" | jq -r '.session_id // empty')
case "$f" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$f" ] || exit 0

case "$f" in
  */segment-anything/*) [ -n "$sid" ] && touch "/tmp/claude-wsam-pyedit-${sid}" ;;
esac

out=$(python3 -m py_compile "$f" 2>&1) && exit 0
echo "py_compile 失敗 ($f): $out" >&2
exit 2
