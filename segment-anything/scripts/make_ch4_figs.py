"""產生 Chapter 4 結果圖：ACDC 逐條件 mIoU 長條圖、注入閘控/餘弦訓練曲線。"""
import csv, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/home/rvl1421/Downloads/figures'
AB = os.path.join(os.path.dirname(__file__), '..', 'outputs_ablation')
os.makedirs(OUT, exist_ok=True)

# Fig 1: ACDC per-condition mIoU bar chart
e = json.load(open(os.path.join(AB, 'R7_seed42', 'e1_results.json')))
pc = e['per_condition_miou']; ov = e['overall_miou'] * 100
conds = ['Fog', 'Rain', 'Snow', 'Night']
vals = [pc['fog'] * 100, pc['rain'] * 100, pc['snow'] * 100, pc['night'] * 100]
colors = ['#4C9F70', '#4C72B0', '#9C7CB8', '#C44E52']
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(conds, vals, color=colors, width=0.62, edgecolor='black', linewidth=0.5)
ax.axhline(ov, ls='--', color='gray', lw=1.2, label='Overall %.2f' % ov)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.8, '%.2f' % v, ha='center', va='bottom',
            fontsize=11, fontweight='bold')
ax.set_ylabel('mIoU (%)', fontsize=12); ax.set_ylim(0, 85)
ax.set_title('ACDC validation mIoU by condition', fontsize=12)
ax.legend(fontsize=10); ax.grid(axis='y', ls=':', alpha=0.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'ch4_acdc_conditions.png'), dpi=200); plt.close()
print('saved ch4_acdc_conditions.png', ['%s %.2f' % (c, v) for c, v in zip(conds, vals)])

# Fig 2: injection gate & cosine training curves (per stage s0-s3)
rows = list(csv.DictReader(open(os.path.join(AB, 'R7_seed42', 'train_log.csv'))))
ep = [int(r['epoch']) for r in rows]
col = lambda n: [float(r[n]) for r in rows]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
cmap = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
for s in range(4):
    a1.plot(ep, col('val_inject_gate_s%d' % s), color=cmap[s], lw=1.6, label='g_s%d' % s)
a1.set_xlabel('Epoch'); a1.set_ylabel('Injection gate value'); a1.set_title('(a) Injection gate')
a1.legend(fontsize=10, ncol=2); a1.grid(ls=':', alpha=0.5)
for s in range(4):
    a2.plot(ep, col('val_inject_cos_s%d' % s), color=cmap[s], lw=1.6, label='stage s%d' % s)
a2.set_xlabel('Epoch'); a2.set_ylabel('cos(q, q + g*delta)'); a2.set_title('(b) Injection cosine similarity')
a2.legend(fontsize=10, ncol=2); a2.grid(ls=':', alpha=0.5); a2.set_ylim(0.6, 1.0)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'ch4_gates_cosine.png'), dpi=200); plt.close()
print('saved ch4_gates_cosine.png final:',
      {'s%d' % s: (round(col('val_inject_gate_s%d' % s)[-1], 3), round(col('val_inject_cos_s%d' % s)[-1], 3)) for s in range(4)})
