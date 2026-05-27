#!/usr/bin/env bash
# 将报告引用的图片统一复制到 report/figures/（组员只需 clone 仓库 + 编译 report）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIG="$ROOT/report/figures"
mkdir -p "$FIG"

cp "$ROOT/badcase_runs/a1_vs_a2/prefix_buckets/chart_doc_len.png" \
   "$FIG/prefix_bucket_doc_len.png"
cp "$ROOT/badcase_runs/a1_vs_a2/prefix_buckets/chart_pos_in_long_doc.png" \
   "$FIG/prefix_bucket_pos_in_long_doc.png"
cp "$ROOT/badcase_runs/a1_vs_a2/prefix_buckets/chart_fixed_prefix.png" \
   "$FIG/prefix_bucket_fixed_prefix.png"

cp "$ROOT/badcase_runs/run3_newdata_newmodel/attn_viz_all/ledger_1/curves_diagnosis.png" \
   "$FIG/attn_run3_ledger1_curves_baseline.png"
cp "$ROOT/badcase_runs/run3_newdata_newmodel/attn_viz_all/ledger_8/heatmap_layer7.png" \
   "$FIG/attn_ledger8_heatmap_baseline.png"
cp "$ROOT/badcase_runs/run3_newdata_newmodel/attn_viz_all/ledger_8/curves_diagnosis.png" \
   "$FIG/attn_ledger8_curves_baseline.png"
cp "$ROOT/badcase_runs/baseline_vs_a1/a1/attn_viz_all/ledger_8/heatmap_layer7.png" \
   "$FIG/attn_ledger8_heatmap_a1.png"
cp "$ROOT/badcase_runs/baseline_vs_a1/a1/attn_viz_all/ledger_8/curves_diagnosis.png" \
   "$FIG/attn_ledger8_curves_a1.png"

echo "已同步到 $FIG"
ls -1 "$FIG"/*.png
