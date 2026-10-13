#!/usr/bin/env bash
# [XYZ-ASSOC] last-step main rows for every arm, then the nine-column table. Run only when the
# GPU given is idle (latency column). Usage: finalize_rows.sh <gpu>
set -uo pipefail
cd /data1/lyq/code/mesh/EventHands1
GPU=${1:-2}
PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
export PYTHONUNBUFFERED=1
mkdir -p outputs/xa/step5
row() {   # run seeds
  echo "[$(date +%F' '%T)] main row $1 seeds $2" | tee -a outputs/xa/step5/finalize.log
  CUDA_VISIBLE_DEVICES=$GPU $PY tools/make_s36_row.py --run "$1" --seeds "$2" --step last \
      > "logs/row_$1_last.log" 2>&1
  echo "[$(date +%F' '%T)] main row $1 rc=$?" | tee -a outputs/xa/step5/finalize.log
}
row xa_c1 3407,3408
row xa_c1r 3407
row xa_c0 3407,3408
row xa_orc 3407,3408
if ls outputs/semkine/xa_ctr_s3408/*step=6000.ckpt > /dev/null 2>&1; then row xa_ctr 3407,3408; fi
$PY tools/report_table.py xa_c1 xa_c1r xa_c0 xa_orc $(ls outputs/semkine/xa_ctr_main_row_last.json >/dev/null 2>&1 && echo xa_ctr) \
    --tag last \
    --label xa_c1="XYZ C1 复现（末步，hit_first，自身 prev 关联）" \
    --label xa_c1r="XYZ C1 同配置同种子复跑（噪声底）" \
    --label xa_c0="XYZ C0 匹配预算（显式 3D 通道置零）" \
    --label xa_orc="XYZ 真值分配上界（分配用窗末真值网格，几何用自身 prev；诊断，不可部署）" \
    --label xa_ctr="XYZ 替代关联（遮挡轮廓双假设 + 近层背面过滤）" \
    | tee outputs/xa/step5/main_table_last.md
echo "FINALIZE_DONE" | tee -a outputs/xa/step5/finalize.log
