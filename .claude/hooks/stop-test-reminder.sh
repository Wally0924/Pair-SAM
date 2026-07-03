#!/usr/bin/env bash
# Stop hook — 本回合改過 segment-anything 的 .py 但沒跑過 pytest 時,提醒使用者(不阻擋)。
input=$(cat)
sid=$(printf '%s' "$input" | jq -r '.session_id // empty')
[ -z "$sid" ] && exit 0
flag="/tmp/claude-wsam-pyedit-${sid}"
if [ -f "$flag" ]; then
  rm -f "$flag"
  echo '{"systemMessage":"提醒:本次修改過 segment-anything 的 .py 檔,尚未偵測到 pytest 執行。建議先驗證再視為完成。"}'
fi
exit 0
