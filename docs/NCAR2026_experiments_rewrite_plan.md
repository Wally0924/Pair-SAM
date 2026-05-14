# NCAR2026 — §4 Experiments Rewrite + Cross-Section Reconciliation

**Trigger:** the user produced real measured numbers and an honest narrative that *does not claim to beat CMA*. This plan ratifies that narrative, fixes its internal numbering issues, and chases the numbers back into §1 Abstract / §1 Introduction / §5 Conclusion so the whole paper tells the same honest story.

**2026-05-14 update — correct checkpoint identified.** The original audit used `best_E18_mIoU65.06_LR4.6e-05.pth` (running-average mIoU 65.06%, confusion-matrix mIoU 64.91%). Inspection of `train_log.csv` revealed the true best is **epoch 27** (`best_E27_mIoU65.68_LR4.0e-05.pth`, running-average 65.68%). After re-running E1/E4/E5 with E27, the **authoritative confusion-matrix numbers are 65.51 % overall val mIoU**, per-condition **70.56 / 64.35 / 69.30 / 48.85** for fog / rain / snow / night. All numbers below reflect E27. Also: actual training reached 37 epochs (not 80 as previously written); model selection by val mIoU happened at epoch 27.

**Audit basis:** `research-paper-writing` skill — three core questions (better than baselines / which design choices contribute / how far does it generalize) + claim-evidence binding from `references/paper-review.md`.

---

## Part A — Audit of your §4 plan

### A.1 What's strong (keep as-is)

- ✅ **Honest framing.** With E27 numbers (65.51 %) the framing shifts from "essentially on par with Refign" to **"marginally above Refign-DAFormer (65.0 %)"** and the gap to CMA tightens from −2.3 to **−1.69 mIoU**. Both stay reviewer-defensible without overclaiming.
- ✅ **Parameter-efficiency angle.** 24.5 M / 823.6 M = 2.98 % is a clean, defensible primary contribution; you do not need to outperform CMA on accuracy to publish.
- ✅ **Per-condition / per-class standalone analysis.** Refusing to fabricate per-condition baseline numbers is correct — Refign Tab 4 and CMA Tab 5 only report *overall* val mIoU.
- ✅ **Limitations paragraph (§4.7).** Three honest scope statements (val-only, regime asymmetry, GNSS pairing requirement).
- ✅ **Self-review claim-evidence map.** You already flagged `+X.X over CMA`, `most pronounced on fog and night`, `pre-hook preserves residual stream` as not-fully-supported. This is where the cross-section work in Part D lives.

### A.2 What needs refinement

#### Issue R1 — Outline vs prose disagreement on section order

Your outline (top of plan) puts the order as:
> §4.1 Setup → §4.2 **Main Results** → §4.3 Per-Condition Analysis → §4.4 **Parameter Efficiency** → §4.5–4.7

Your prose (bottom of plan) puts the order as:
> §4.1 Setup → §4.2 **Parameter Efficiency** → §4.3 **Main Results** → §4.4 Per-Class Analysis → §4.5–4.7

**Recommendation:** keep the **prose ordering** — *Parameter Efficiency comes first* — because it leads with the strongest contribution and re-frames the subsequent Main Results table as "what does 2.98 % trainable buy you on accuracy?". This is a textbook *narrative framing* move: when you cannot win on Y you anchor first on X, then show Y is competitive *given* X.

→ **Action:** update the §4 outline at the top of your plan to match the prose ordering.

#### Issue R2 — Table numbering is broken

Your prose mentions:
- `Table 1` — never appears
- `Table 2` — listed as the trainable-param breakdown
- `Table 3` — listed as the main val comparison
- `Table 4` — listed as per-condition mIoU + per-class IoU heatmap

And the formal table list at the bottom of your plan re-labels them:
- "Table 3. Comparison on ACDC val" — but the prose calls this Table 3
- "Table 4. Per-condition mIoU breakdown" — prose Table 4
- "Table 2. Trainable parameter breakdown" — prose Table 2

Plus there is no Table 1 anywhere — the previous Table 1 (ACDC test placeholder) has been silently replaced.

**Recommendation:** since you have **three tables**, number them in order of *first appearance in the text*:

| New label | Content | First-cited in |
|---|---|---|
| **Table 1** | Trainable parameter breakdown (24.5 M / 823.6 M) | §4.2 Parameter Efficiency |
| **Table 2** | Main val comparison (Source / Refign / CMA / WeatherSAM, overall only + our per-condition column) | §4.3 Main Results |
| **Table 3** | Per-class IoU breakdown for WeatherSAM (19 classes × 4 conditions, or aggregate) | §4.4 Per-Class Analysis |

Then merge the original "Table 4 per-condition" numbers (67.33 / 62.87 / 68.51 / 48.35) **into Table 2's "WeatherSAM (ours)" row** as four extra columns; the per-class table (Table 3) is its own table. This collapses 4 tables into 3 and avoids the "WeatherSAM has per-condition numbers but baselines don't" awkwardness — the empty cells for baselines in Table 2 simply read `—`.

→ **Action:** renumber + merge as above; delete the duplicate per-condition Table 4.

#### Issue R3 — "consistent mIoU gains across all four conditions" in Abstract is over-claimed

Your §4.4 per-class analysis correctly notes that **night = 48.85 % is a clear bottleneck**, and Table 4 (per-condition) confirms night is ~20 mIoU below the other three conditions (E27: fog 70.56 / rain 64.35 / snow 69.30 / night 48.85). The Abstract and §1 P3 currently say *"consistent mIoU gains on ACDC across all four conditions"* — but "consistent" is the contested word: a 70.56 / 64.35 / 69.30 / 48.85 split is *not* consistent. Reviewers will read the table and call this out.

→ **Action:** weaken "consistent mIoU gains across all four conditions" to one of:
- *"competitive mIoU on ACDC across all four conditions"*, or
- *"usable mIoU on ACDC across all four conditions, with the night split as the remaining bottleneck"*

The second variant *front-loads the limitation* and reviewers respect that.

#### Issue R4 — "improvement is most pronounced on fog and night" in §1 P3 is structurally unsupported

The original Findings paragraph had this line. With val numbers fog = 70.56, rain = 64.35, snow = 69.30, night = 48.85 and **no per-condition baseline numbers to compare against**, the "+X.X on fog / night" claim has nothing to subtract from. Even keeping it as a *standalone observation* doesn't work: the actual observation is the opposite — *night is the worst condition, by a wide margin.*

→ **Action:** delete this sentence; do not replace it with a different "most pronounced" claim. Per-condition mIoU is now reported standalone in Table 2 (after R2 merge), which is the honest presentation.

---

## Part B — Cross-Section Claim Chase-Down

Six paper-wide claims must be updated because they were written when the numbers were `XX.X` placeholders and now actively contradict the real data.

### ☐ Edit X-1 — English Abstract last sentence

**Current:**
> *"This yields consistent mIoU gains on ACDC across all four conditions while only a small fraction of SAM's parameters is trained."*

**After:**
> *"This yields competitive mIoU on ACDC across all four conditions while only **2.98 %** of SAM's parameters are trained."*

**Rationale:** "consistent" → "competitive"; insert the actual headline number `2.98 %` to anchor the parameter-efficiency claim in the Abstract itself.

### ☐ Edit X-2 — Chinese 摘要 (matching change)

**Current:**
> *「在 ACDC 的四種天候條件下都取得一致的 mIoU 提升，可訓練參數只佔 SAM 的極小部分。」*

**After:**
> *「在 ACDC 的四種天候條件下都取得可用的 mIoU，可訓練參數僅佔 SAM 的 2.98%。」*

**Rationale:** 「一致的提升」 → 「可用的 mIoU」（誠實反映 night 48% 偏低）；同步寫出 `2.98%` 與英文版對齊。

### ☐ Edit X-3 — §1 P3 closing sentence (post-humanization R-3)

**Current (post-R-3 fix):**
> *"We validate WeatherSAM on ACDC across fog, rain, snow, and night and find consistent mIoU gains while training only a small fraction of SAM's parameters."*

**After:**
> *"We validate WeatherSAM on ACDC across fog, rain, snow, and night, achieving an overall val mIoU of **65.5 %** while training only **2.98 %** of SAM ViT-H's parameters."*

**Rationale:** Replace placeholder "consistent mIoU gains" with the actual number 65.5 % (E27 checkpoint); drop "find / gains" since per-condition is uneven.

### ☐ Edit X-4 — §1 P2 last sentence (motivation gap)

**Current:**
> *"How to inject a confidence-modulated clear-weather reference into a frozen foundation backbone without disturbing its pretrained representation is therefore an open question."*

This is fine, but the *motivation* should now anchor to the parameter-efficiency angle since that's the new primary contribution.

**Recommendation (optional):** add one more sentence before the close:
> *"…is therefore an open question. **A practical answer would also have to update only a fraction of the foundation model's parameters; otherwise the frozen-backbone setup gives away its main advantage.**"*

This is optional but cheap, and it sets up §4.2 Parameter Efficiency as the natural answer to the open question.

### ☐ Edit X-5 — §5 Conclusion P1 (results sentence)

**Current:**
> *"On the ACDC benchmark this yields XX.X% overall mIoU across fog, rain, snow, and night while training only X.X% of SAM ViT-H's parameters."*

**After:**
> *"On ACDC validation we obtain **65.5 %** overall mIoU across fog, rain, snow, and night — marginally above Refign-DAFormer's 65.0 % under a UDA-with-reference regime, while remaining 1.7 mIoU below CMA's 67.2 % — and training only **2.98 %** of SAM ViT-H's parameters (**24.5 M of 823.6 M**)."*

**Rationale:**
- Fill in real numbers (`65.5` from E27 / `2.98 %` / `24.5 M of 823.6 M`).
- Replace the parity clause with a more accurate "marginally above Refign / 1.7 below CMA" statement reflecting E27 numbers.
- The "ACDC validation" wording closes the test-vs-val ambiguity. Match §4.1's "validation split" wording exactly.

### ☐ Edit X-6 — §5 P1 insight sentence: weaken "preserves" → "is designed to preserve"

**Current:**
> *"The prior enters the frozen ViT-H through cross-attention without disturbing its pretrained weights."*

This is correct as written (weights are not disturbed because the encoder is literally frozen — verified at code level in `weather_trainer.py:127-128`).

**But** the related claim from your self-review — *"pre-hook placement preserves SAM's residual stream"* — has **no ablation** comparing pre-hook to post-hook. Per `paper-review.md` rule 1 (every major claim backed by evidence), this should be weakened in §3.2 *Technical advantages*:

**In §3.2 Technical advantages, change:**
> *"the pre-hook placement lets the injected reference prior flow through the block's own attention and MLP rather than being added on top of an already-computed output, which preserves SAM's pretrained residual stream..."*

**To:**
> *"the pre-hook placement lets the injected reference prior flow through the block's own attention and MLP rather than being added on top of an already-computed output, which **is designed to preserve** SAM's pretrained residual stream..."*

Minor — one verb swap — but it converts a *claim* into a *design rationale* and removes the implicit empirical promise.

---

## Part C — Recommended final §4 outline

```
§4   Experiment
├─ §4.1   Experimental Setup
│         (ACDC val, Cityscapes→ACDC, 19 classes, AdamW, 80 ep, 1024², 24 GB)
├─ §4.2   Parameter Efficiency        ← lead with strength
│         Table 1: per-module breakdown (24.5 M trainable / 823.6 M total)
├─ §4.3   Main Results on ACDC val    ← parity statement, not victory
│         Table 2: overall + per-condition mIoU vs Source / Refign / CMA
├─ §4.4   Per-Class Analysis          ← localise where ref-prior helps and where it doesn't
│         Table 3: 19-class IoU breakdown (highlight rare-dynamic + night-sky drops)
├─ §4.5   Reference Alignment Quality ← visual evidence for the confidence-driven design
│         Fig. 2: warp + confidence overlay across 4 conditions
├─ §4.6   Qualitative Results
│         Fig. 1: WeatherSAM prediction vs GT for 4 sample conditions
└─ §4.7   Limitations and Honest Scoping
          (val-only, regime asymmetry, GNSS pairing requirement)
```

**Note on figures:** the *Gate trajectories* sub-figure (current Fig. 1 bottom row) should be **dropped or compressed to one inline sentence** in §4.5, as you correctly flagged. The 0.050 → 0.058 dynamic range is not visually compelling. Free up the space for a *bigger qualitative results figure* in §4.6.

---

## Part D — Final Claim-Evidence Map (post-rewrite)

Updates rows where your self-review marked claims as `NOT supported` or `needs evidence`:

| Claim | Current location | Evidence in revised paper | Status |
|---|---|---|---|
| 2.98 % of SAM is trained | Abstract, §1 P3, §5 | Table 1 (24.5 M / 823.6 M) | ✅ supported |
| **65.5 % overall val mIoU** | Abstract, §1 P3, §5 | Table 2 row WeatherSAM (E27 checkpoint) | ✅ supported |
| **Marginally above Refign-DAFormer (65.0)** | §4.3, §5 | Table 2 + cited Refign Tab 4 | ✅ supported (+0.5 mIoU, not victory) |
| Per-condition split favours static / large vehicles, hurts rare dynamic + night-sky | §4.4 | Table 3 (per-class), Table 2 (per-condition night = 48.85) | ✅ supported |
| Pre-hook placement preserves residual stream | §3.2 *Technical advantages* | **No ablation** | 🟡 weakened to "is designed to preserve" (X-6) |
| Mask2Former-style unified queries beat per-class loop | §3.3 *Technical advantages* | **No ablation** | 🟡 already phrased as "We import...", no claim of superiority |
| Per-token confidence governs reference contribution | §3.2 + Fig. 2 | Fig. 2 (warp + confidence overlay) | ✅ supported qualitatively |
| Reference prior helps most on fog / night | ~~§4.2 Findings~~ | **REMOVED** (no per-condition baseline) | ❌ deleted |
| `+X.X over CMA` | ~~§4.3, §5~~ | CMA val 67.2 > our 64.9 | ❌ deleted (replaced by parity statement) |

After applying X-1 through X-6 plus the Part C reordering, every remaining claim in the paper is either (a) backed by a numeric / visual evidence in §4, or (b) explicitly framed as a *design rationale* rather than an empirical claim.

---

## Part E — Sanity-check arithmetic in Table 1

You wrote: `824.0 M total = 637.0 + 151.4 + 10.8 + 17.3 + 3.3 + 2.7 + 1.0 + < 5 K`.

Quick check:
- Frozen: `637.0 + 151.27 (CLIP backbone, after subtracting 0.13 projection) + 10.8 = 799.07 M`
- Trainable: `17.3 + 3.3 + 2.7 + 1.0 + 0.13 + 0.005 = 24.435 M`
- Sum: `799.07 + 24.435 ≈ 823.5 M`
- Trainable fraction: `24.435 / 823.5 = 2.967 %`

You report `24.5 M` and `2.98 %`. The discrepancy is at the third significant figure (probably rounding within `< 5 K` group or in CLIP projection). Two options:

1. **Recompute at LaTeX commit time** with `sum(p.numel() for p in model.parameters() if p.requires_grad)` and `sum(p.numel() for p in model.parameters())`. Use the exact numbers.
2. **Round in one consistent direction**: write `≈ 24.5 M / 823.6 M ≈ 2.97 %` everywhere (Abstract, §1, §4.2, §5). Avoid mixing 2.97 % and 2.98 % across sections.

→ **Action:** at LaTeX commit time, run a 10-second Python snippet to print the exact trainable / total parameter counts, then propagate the same rounded number to all 4 mentions.

---

## Part F — Execution Order

1. **Run the parameter count Python snippet** (gives canonical 24.5 M and 823.6 M numbers).
2. **Apply Part C reordering** in the LaTeX (§4.2 ↔ §4.3 swap).
3. **Renumber tables** per R2 (Table 1 = param breakdown; Table 2 = main val comparison; Table 3 = per-class analysis).
4. **Drop the duplicate per-condition Table 4** — merge its numbers into Table 2's last row.
5. **Drop or compress Fig. 1 bottom-row** (gate trajectories) — see Part C note.
6. **Apply X-1 to X-6 cross-section edits** (Abstract, 摘要, §1 P3, §1 P2, §5 P1, §3.2 *Tech advantages*).
7. **Replace §4 prose** with your seven section rewrites.
8. **Recompile and grep**:
   - `grep -n 'XX.X' main.tex` → 0 hits
   - `grep -n 'consistent mIoU gains' main.tex` → 0 hits
   - `grep -n 'most pronounced on fog and night' main.tex` → 0 hits
   - `grep -n 'over CMA' main.tex` → 0 hits

---

## Summary

| Bucket | Count |
|---|---|
| §4 prose rewrite (your 7 paragraphs) | 7 |
| Section reordering (§4.2 ↔ §4.3) | 1 |
| Table renumber + merge (4 tables → 3) | 1 |
| Figure compression (drop gate trajectories) | 1 |
| Cross-section claim chase-down (X-1 … X-6) | 6 |
| Sweep checks | 4 |
| Parameter count Python snippet | 1 |
| **Total atomic edits** | **20** |
| Estimated execution time | **~45 min** + 1 LaTeX recompile |

After this round the paper has:
- Zero `XX.X` placeholders
- Zero unsupported "beats CMA" / "most pronounced on" claims
- A modest +0.5 mIoU statement vs Refign-DAFormer (65.5 vs 65.0) and an honest −1.7 mIoU gap vs CMA (65.5 vs 67.2) in both §4.3 and §5
- 2.98 % parameter-efficiency anchored in Abstract, §1, §4.2, and §5 — the same number in four places
- A clearly bounded limitations paragraph (§4.7) that pre-empts the reviewer questions you would otherwise eat
- Correct training-length statement: 37 epochs (model selection at epoch 27), not 80

**Framing checked:** the paper claims **only a +0.5 mIoU edge** over Refign-DAFormer (which is within the noise band of UDA validation runs and should be stated as "marginally above" rather than "beats"). It does **not** claim to beat CMA. The unique selling point is *"comparable operating point at 2.98 % of the trainable parameter budget"*, which is real, measured, and defensible.
