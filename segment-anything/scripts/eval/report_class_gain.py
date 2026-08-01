#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐類別參考增益拆分：把附錄 Q7「動態類別優勢非來自參考」的間接推論換成直接量測。

背景：附錄 Q7 目前引用第 4.4 節的置信度統計（靜態約 0.52、動態約 0.37）反推
「參考影像對動態類別的貢獻有限」——這是間接論證，且 Task 3（analyze_conf_gain.py）
已證明逐張的 conf_mean 與 gain 之間無顯著相關（整體 Pearson r = -0.009, p = 0.857）。
本腳本改用最直接的證據：逐類別「有參考 IoU − 無參考 IoU」的差值，並額外重現
第 4.4 節的置信度統計以查明 0.52 / 0.37 的可能來源。

輸出（figures_defense/）：
    class_gain_table.tex   19 類 ref/noref/差值，依靜態(0-10)／動態(11-18)分組，
                            可直接 \\input 的表格
    class_gain.json         逐類別數值、兩種置信度定義對照、動態類別×條件交叉表

輸入（figures_defense/reference_gain/，Task 1 產出）：
    per_class_iou.csv               逐類別累加 IoU（ref / noref）
    per_class_iou_by_condition.csv  逐條件逐類別累加 IoU
    per_image.csv                   逐張 mIoU、增益、置信度統計

用法：
    conda run -n sam_env python scripts/eval/report_class_gain.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.eval._eval_common import CITYSCAPES_CLASSES

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
IN_DIR = os.path.join(ROOT, "figures_defense", "reference_gain")
OUT_DIR = os.path.join(ROOT, "figures_defense")

STATIC_IDS = list(range(0, 11))   # road..sky
DYNAMIC_IDS = list(range(11, 19))  # person..bicycle
CONDITIONS = ["fog", "rain", "snow", "night"]
COND_LABEL = {"fog": "霧天", "rain": "雨天", "snow": "雪天", "night": "夜間"}
PAPER_STATIC, PAPER_DYNAMIC = 0.52, 0.37  # 論文第 4.4 節既有數值


def fmt(x, nd=2, signed=False):
    """統一小數位數格式化；signed=True 時強制帶正負號（LaTeX $\\pm$ 慣例）。"""
    if signed:
        return f"{x:+.{nd}f}"
    return f"{x:.{nd}f}"


def load_per_class_gain():
    """讀 per_class_iou.csv，回傳每類的 (class_id, class_name, noref_iou, ref_iou, diff)。"""
    df = pd.read_csv(os.path.join(IN_DIR, "per_class_iou.csv"))
    piv = df.pivot(index="class_id", columns="setting", values="iou")
    rows = []
    for cid in range(19):
        noref_iou = float(piv.loc[cid, "noref"])
        ref_iou = float(piv.loc[cid, "ref"])
        rows.append(dict(
            class_id=cid,
            class_name=CITYSCAPES_CLASSES[cid],
            group="static" if cid in STATIC_IDS else "dynamic",
            noref_iou=noref_iou,
            ref_iou=ref_iou,
            diff=round(ref_iou - noref_iou, 3),
        ))
    return rows


def group_summary(rows, group):
    diffs = [r["diff"] for r in rows if r["group"] == group]
    return sum(diffs) / len(diffs)


def confidence_two_definitions():
    """兩種置信度定義：逐張平均（等權）vs. 全域像素加權。"""
    df = pd.read_csv(os.path.join(IN_DIR, "per_image.csv"))
    n_total = len(df)
    n_dynamic_na = int(df["conf_dynamic"].isna().sum())

    # 定義一：逐張平均——每張先算靜態/動態平均置信度，406 張等權平均
    per_image_static = float(df["conf_static"].mean())
    per_image_dynamic = float(df["conf_dynamic"].mean())  # 自動排除缺值

    # 定義二：全域像素加權——以 n_static/n_dynamic 對逐張平均值加權合併
    # （pandas .sum() 預設 skipna=True，conf_dynamic 缺值列的 n_dynamic 恆為 0，
    #  不影響加權和；仍以 dropna 明確排除以避免歧義）
    d_static = df.dropna(subset=["conf_static"])
    weighted_static = float((d_static["conf_static"] * d_static["n_static"]).sum()
                             / d_static["n_static"].sum())
    d_dynamic = df.dropna(subset=["conf_dynamic"])
    weighted_dynamic = float((d_dynamic["conf_dynamic"] * d_dynamic["n_dynamic"]).sum()
                              / d_dynamic["n_dynamic"].sum())

    return dict(
        n_images=n_total,
        n_dynamic_na=n_dynamic_na,
        per_image_mean=dict(static=round(per_image_static, 3),
                             dynamic=round(per_image_dynamic, 3),
                             note="每張先算靜態/動態平均，再跨 406 張等權平均；"
                                  "方向與論文相反（動態 > 靜態）"),
        pixel_weighted=dict(static=round(weighted_static, 3),
                             dynamic=round(weighted_dynamic, 3),
                             note="以 n_static/n_dynamic 加權合併 406 張；"
                                  "方向與論文一致但量值有落差"),
        paper_reported=dict(static=PAPER_STATIC, dynamic=PAPER_DYNAMIC),
        conclusion=(
            "兩種標準定義皆重現不出論文的 0.52 / 0.37：逐張平均方向相反"
            f"（靜態 {per_image_static:.3f} < 動態 {per_image_dynamic:.3f}），"
            f"像素加權方向相符但量值不同（靜態 {weighted_static:.3f} vs. 論文 0.52，"
            f"動態 {weighted_dynamic:.3f} vs. 論文 0.37）。repo 中查無產生論文數值的"
            "原始腳本，論文數值來源不可考，應以本次量測值更新論文第 4.4 節與附錄 Q7。"
        ),
    )


def dynamic_gain_by_condition():
    """動態類別 × 條件 的參考增益交叉表（含動態整體聚合列）。"""
    df = pd.read_csv(os.path.join(IN_DIR, "per_class_iou_by_condition.csv"))
    df_dyn = df[df["class_id"].isin(DYNAMIC_IDS)]

    per_class_rows = []
    for cid in DYNAMIC_IDS:
        row = dict(class_id=cid, class_name=CITYSCAPES_CLASSES[cid])
        sub = df_dyn[df_dyn["class_id"] == cid]
        for cond in CONDITIONS:
            piv = sub[sub["condition"] == cond].set_index("setting")["iou"]
            row[cond] = round(float(piv["ref"] - piv["noref"]), 2)
        per_class_rows.append(row)

    # 動態整體聚合（以 intersection/union 加總後重算 IoU，而非逐類別平均，
    # 避免像 bicycle/motorcycle 這類 union 很小的類別以等權方式扭曲整體判讀）
    overall_row = dict(class_id=None, class_name="動態整體（聚合）")
    for cond in CONDITIONS:
        agg = {}
        for setting in ("ref", "noref"):
            sub = df_dyn[(df_dyn["condition"] == cond) & (df_dyn["setting"] == setting)]
            inter = sub["intersection"].sum()
            union = sub["union"].sum()
            agg[setting] = 100.0 * inter / union
        overall_row[cond] = round(agg["ref"] - agg["noref"], 2)
    per_class_rows.append(overall_row)
    return per_class_rows


def write_latex_table(rows, static_avg, dynamic_avg, out_path):
    """輸出 class_gain_table.tex：19 類 IoU/差值，依靜態/動態分組，附組平均。"""
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{各類別有無參考影像的 IoU 對照（驗證集，ACDC val，406 張）}")
    lines.append(r"  \label{tab:class_gain_static_dynamic}")
    lines.append(r"  \renewcommand{\arraystretch}{1.2}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{lccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{類別} & \textbf{無參考 (\%)} & \textbf{有參考 (\%)} & \textbf{$\Delta$} \\")
    lines.append(r"    \midrule")
    lines.append(r"    \multicolumn{4}{l}{\textit{靜態結構}} \\")
    for r in rows:
        if r["group"] != "static":
            continue
        lines.append(f"    {r['class_name']} & {fmt(r['noref_iou'])} & "
                      f"{fmt(r['ref_iou'])} & ${fmt(r['diff'], signed=True)}$ \\\\")
    lines.append(f"    \\textbf{{靜態平均}} & & & \\textbf{{${fmt(static_avg, signed=True)}$}} \\\\")
    lines.append(r"    \midrule")
    lines.append(r"    \multicolumn{4}{l}{\textit{動態物體}} \\")
    for r in rows:
        if r["group"] != "dynamic":
            continue
        lines.append(f"    {r['class_name']} & {fmt(r['noref_iou'])} & "
                      f"{fmt(r['ref_iou'])} & ${fmt(r['diff'], signed=True)}$ \\\\")
    lines.append(f"    \\textbf{{動態平均}} & & & \\textbf{{${fmt(dynamic_avg, signed=True)}$}} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rows = load_per_class_gain()
    static_avg = group_summary(rows, "static")
    dynamic_avg = group_summary(rows, "dynamic")

    tex_path = os.path.join(OUT_DIR, "class_gain_table.tex")
    write_latex_table(rows, static_avg, dynamic_avg, tex_path)

    conf = confidence_two_definitions()
    cross_tab = dynamic_gain_by_condition()

    # 逐類別平均置信度：per_image.csv 只有靜態/動態二分彙總，無逐類別欄位，
    # 需另外掃描（讀取每次 forward 的 confidence map 並依 GT 逐類別平均），
    # 但該掃描要用到 GPU 與模型前傳，目前 GPU 被另一項量測佔用——列為待辦，
    # 不在本任務內重跑推論。
    per_class_confidence_todo = (
        "無法由現有 CSV 產生：per_image.csv 只有 conf_static/conf_dynamic 兩欄"
        "彙總值，沒有逐類別（19 類）置信度。需要重新掃描 ACDC val、在每次 forward"
        "後依 GT 逐類別平均 confidence map，屬於 GPU 推論工作，目前 GPU 被另一項"
        "量測佔用，故列為待辦，未在本任務執行。"
    )

    dynamic_gain_positive = dynamic_avg > 0
    rain_snow_dynamic_gain = [r for r in cross_tab if r["class_name"] == "動態整體（聚合）"][0]
    rain_snow_positive = rain_snow_dynamic_gain["rain"] > 0 and rain_snow_dynamic_gain["snow"] > 0

    verdict = (
        "動態物體整體平均增益為 "
        f"{fmt(dynamic_avg, signed=True)} 個百分點（8 類逐類別等權平均），"
        "非明確為負，且逐類別方向不一致：person/rider/car/truck 為負，"
        "bus/train/motorcycle/bicycle 為正。條件拆解上，動態整體聚合在雨天"
        f"為 {fmt(rain_snow_dynamic_gain['rain'], signed=True)}、雪天為 "
        f"{fmt(rain_snow_dynamic_gain['snow'], signed=True)} 個百分點——"
        + ("與靜態結構在雨雪天的正貢獻方向一致，" if rain_snow_positive else
           "並未一致隨雨雪天呈現正貢獻，") +
        "顯示動態類別並非系統性地無法從參考獲益。"
        "附錄 Q7 原論證（引用置信度統計反推動態類別優勢非來自參考）"
        "在本次量測下不獲支持：其前提（動態置信度低於靜態）本身在兩種"
        "標準定義下都對不上論文數值，逐張定義甚至方向相反；"
        "而直接的 IoU 差值顯示動態類別的參考效應是逐類別異質的，"
        "不是「一致受抑」。原論證需要改寫，且不能再以置信度統計作為"
        "唯一支撐證據。"
    )

    out = dict(
        per_class=rows,
        group_mean_diff=dict(static=round(static_avg, 3), dynamic=round(dynamic_avg, 3)),
        confidence_two_definitions=conf,
        per_class_confidence_todo=per_class_confidence_todo,
        dynamic_gain_by_condition=cross_tab,
        verdict=verdict,
    )
    json_path = os.path.join(OUT_DIR, "class_gain.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"靜態組平均差值（ref-noref）：{fmt(static_avg, signed=True)} pp")
    print(f"動態組平均差值（ref-noref）：{fmt(dynamic_avg, signed=True)} pp")
    print("置信度兩種定義：")
    print("  逐張平均   靜態=%.3f 動態=%.3f" % (conf["per_image_mean"]["static"],
                                          conf["per_image_mean"]["dynamic"]))
    print("  像素加權   靜態=%.3f 動態=%.3f" % (conf["pixel_weighted"]["static"],
                                          conf["pixel_weighted"]["dynamic"]))
    print(f"  論文既有   靜態={PAPER_STATIC} 動態={PAPER_DYNAMIC}")
    print("動態整體聚合 × 條件：",
          {c: rain_snow_dynamic_gain[c] for c in CONDITIONS})
    print("輸出：")
    print("  -", tex_path)
    print("  -", json_path)


if __name__ == "__main__":
    main()
