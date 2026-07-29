#!/usr/bin/env bash
# SessionStart hook — 注入當前實驗快照(branch、最近 commits、未提交變更)。
cd /home/rvl1421/SAM_research-1 2>/dev/null || exit 0
echo "[PairSAM 實驗快照]"
echo "branch: $(git branch --show-current 2>/dev/null)"
echo "最近 commits:"
git log -3 --oneline 2>/dev/null
changes=$(git status --porcelain 2>/dev/null | head -10)
if [ -n "$changes" ]; then
  echo "未提交變更:"
  echo "$changes"
fi
exit 0
