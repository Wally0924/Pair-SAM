# v15 (E18) 權重評估實驗產出 — 2026-05-14

**Checkpoint:** `best_E18_mIoU65.06_LR4.6e-05.pth`

| 實驗 | 產出 | 對應論文 |
|------|------|----------|
| E1 — ACDC val 完整評估 | [`e1_acdc_val_results.md`](e1_acdc_val_results.md), [`e1_acdc_val_results.json`](e1_acdc_val_results.json) | Refign Tab.1 / CMA Tab.1 |
| E4 — 定性比較圖 | [`e4_qualitative.png`](e4_qualitative.png) | Refign Fig.4 / CMA Fig.4 |
| E5 — UAWarpC warp + confidence | [`e5_warp_confidence.png`](e5_warp_confidence.png) | Refign Fig.7 |

## 重現方式

```bash
cd /home/rvl1421/SAM_research-1
conda run -n sam_env python segment-anything/scripts/eval/eval_e1_acdc_val_full.py
conda run -n sam_env python segment-anything/scripts/eval/viz_e4_qualitative.py
conda run -n sam_env python segment-anything/scripts/eval/viz_e5_warp_confidence.py
```
