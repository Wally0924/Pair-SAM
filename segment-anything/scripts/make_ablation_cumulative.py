"""§4.6 累積消融折線圖：R1->R7 模組逐項加入的 mIoU 軌跡（凸顯 R3->R4 非單調下凹）。
主軌跡藍色；落差以紅色垂直箭頭標於 R4 上方空白帶；per-class baseline 標籤置於圖框外右側留白。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/home/rvl1421/Downloads/Images'
BLUE = '#2A6FB0'
RED = '#C0392B'
BLACK = '#1A1A1A'
mious = [39.6, 60.7, 62.8, 41.8, 54.8, 61.2, 67.26]
adds = ['Baseline\n(frozen SAM)', '+Ref', '+Pre-inject', '+Unified\nquery',
        '+LRH', '+Lov.+Dice', '+MFB\n(Full)']
x = list(range(7))

fig, ax = plt.subplots(figsize=(9.6, 5.6))
fig.subplots_adjust(left=0.085, right=0.965, top=0.90, bottom=0.14)
ax.set_xlim(-0.45, 6.45)
ax.set_ylim(35, 75)

# per-class 基準（黑色虛線；說明由圖說 caption 提供，圖內不另標文字）
ax.axhline(62.8, ls='--', color=BLACK, lw=1.3, alpha=0.55, zorder=1)

# 主軌跡（藍）
ax.plot(x, mious, '-o', color=BLUE, lw=2.8, ms=12, zorder=3)

# 各點數值（黑、加粗）；谷底 R4 放下方，其餘放上方
offset = {0: (0, 12, 'bottom'), 1: (-2, 12, 'bottom'), 2: (-6, 12, 'bottom'),
          3: (0, -20, 'top'), 4: (-4, 12, 'bottom'), 5: (0, 12, 'bottom'),
          6: (0, 12, 'bottom')}
for xi, m in zip(x, mious):
    dx, dy, va = offset[xi]
    ax.annotate('%.1f' % m, (xi, m), textcoords='offset points', xytext=(dx, dy),
                ha='center', va=va, fontsize=13, fontweight='bold', color=BLACK)

# 落差：紅色垂直箭頭，置於 R4 正上方空白帶（不與折線重疊）
ax.annotate('', xy=(3, 43.6), xytext=(3, 62.4),
            arrowprops=dict(arrowstyle='-|>', color=RED, lw=2.6, mutation_scale=24), zorder=4)
ax.text(3, 65.0, r'$-21.0$', fontsize=16, fontweight='bold', color=RED,
        ha='center', va='bottom')

ax.set_xticks(x); ax.set_xticklabels(adds, fontsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.set_ylabel('Overall mIoU (%)', fontsize=14)
ax.set_title('Cumulative ablation: module-by-module mIoU (ACDC val)', fontsize=15, pad=12)
ax.grid(axis='y', ls=':', alpha=0.45)
plt.savefig(OUT + '/ch4_ablation_cumulative.png', dpi=200)
plt.close()
print('saved', OUT + '/ch4_ablation_cumulative.png')
