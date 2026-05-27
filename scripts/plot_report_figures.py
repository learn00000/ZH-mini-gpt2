#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成中期报告用图：
  1. report/figures/genre_ratio.png       — manifest 四体裁占比
  2. report/figures/badcase_run3_by_src.png — Run3 badcase 按 src 分桶

用法（仓库根目录）：
  python scripts/plot_report_figures.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
SUMMARY_PATH = ROOT / "badcase_runs/run3_newdata_newmodel/badcase_summary.json"
LEDGER_PATH = ROOT / "badcase_runs/run3_newdata_newmodel/badcase_ledger.jsonl"

# record.txt：清洗后全量 manifest 体裁占比（若无 summary 则回退）
MANIFEST_GENRE_RATIO = {
    "news": 0.42,
    "comment": 0.25,
    "webtext": 0.23,
    "wiki": 0.11,
}

GENRE_LABELS = {
    "news": "news（新闻）",
    "comment": "comment（评论）",
    "webtext": "webtext（网文）",
    "wiki": "wiki（百科）",
}

SRC_ORDER = ["news", "comment", "webtext", "wiki"]
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]


FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def setup_cjk_font() -> None:
    if Path(FONT_PATH).is_file():
        fm.fontManager.addfont(FONT_PATH)
        name = fm.FontProperties(fname=FONT_PATH).get_name()
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_genre_counts(summary_path: Path) -> dict[str, int]:
    if not summary_path.is_file():
        return {}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    counts = data.get("raw_type_field_counts") or data.get("counts_by_norm_type_in_file")
    if not counts:
        return {}
    return {k: int(v) for k, v in counts.items() if k in SRC_ORDER}


def plot_genre_ratio(out_path: Path, summary_path: Path) -> None:
    counts = load_genre_counts(summary_path)
    if counts:
        total = sum(counts.get(k, 0) for k in SRC_ORDER)
        ratios = {k: counts.get(k, 0) / total for k in SRC_ORDER}
        subtitle = f"依据 test.jsonl 行数统计（N={total:,}）"
    else:
        ratios = MANIFEST_GENRE_RATIO
        subtitle = "依据 record.txt 全量 manifest 统计（约 7606 万行）"

    labels = [GENRE_LABELS[k] for k in SRC_ORDER]
    sizes = [ratios[k] * 100 for k in SRC_ORDER]
    explode = (0.02, 0.02, 0.02, 0.02)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=COLORS,
        autopct="%1.1f%%",
        startangle=90,
        explode=explode,
        pctdistance=0.75,
        textprops={"fontsize": 10},
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("清洗后 manifest 四体裁占比\n" + subtitle, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"已写 {out_path}")


def classify_row(row: dict) -> str:
    flags = row.get("heuristic_flags") or {}
    if flags.get("repeated_span_24"):
        return "rep24"
    if flags.get("repeated_span_12"):
        return "rep12"
    if flags.get("any_issue_hint"):
        return "other"
    if flags.get("low_diversity"):
        return "low_div"
    return "ok"


def plot_badcase_run3(ledger_path: Path, out_path: Path) -> None:
    rows = []
    with ledger_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_src: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        src = row.get("sample_type") or row.get("raw_type_field") or "unknown"
        by_src[src][classify_row(row)] += 1

    categories = ["ok", "rep12", "rep24", "low_div", "other"]
    cat_labels = {
        "ok": "无强提示",
        "rep12": "rep12",
        "rep24": "rep24",
        "low_div": "低多样性",
        "other": "其它提示",
    }
    cat_colors = {
        "ok": "#BAB0AC",
        "rep12": "#F58518",
        "rep24": "#E45756",
        "low_div": "#72B7B2",
        "other": "#B279A2",
    }

    x = range(len(SRC_ORDER))
    width = 0.55
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    bottom = [0] * len(SRC_ORDER)
    for cat in categories:
        vals = [by_src.get(src, Counter()).get(cat, 0) for src in SRC_ORDER]
        if sum(vals) == 0:
            continue
        ax.bar(
            x,
            vals,
            width,
            bottom=bottom,
            label=cat_labels[cat],
            color=cat_colors[cat],
            edgecolor="white",
            linewidth=0.6,
        )
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xticks(list(x))
    ax.set_xticklabels([GENRE_LABELS[s] for s in SRC_ORDER], fontsize=10)
    ax.set_ylabel("badcase 条数（每体裁 8 条）")
    ax.set_title(
        "Run3 badcase 按 src 分布\n"
        f"ckpt: 20260519_195338 · manifest test · 共 {len(rows)} 条",
        fontsize=11,
    )
    ax.set_ylim(0, 8.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    # 标注每柱总数
    for i, src in enumerate(SRC_ORDER):
        total = sum(by_src.get(src, Counter()).values())
        ax.text(i, total + 0.15, str(total), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"已写 {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成中期报告 figures")
    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_PATH,
        help="badcase_summary.json 路径",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=LEDGER_PATH,
        help="badcase_ledger.jsonl 路径",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIG_DIR,
        help="输出目录",
    )
    args = parser.parse_args()

    setup_cjk_font()
    plot_genre_ratio(args.out_dir / "genre_ratio.png", args.summary)
    plot_badcase_run3(args.ledger, args.out_dir / "badcase_run3_by_src.png")


if __name__ == "__main__":
    main()
