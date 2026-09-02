#!/usr/bin/env bash
# 在你的 Ubuntu 电脑上运行,从训练服务器拉取"复现 15mm local 误差"所需的最小集合。
# 用法:
#   bash pull_on_ubuntu.sh <服务器SSH地址>
#   例:  bash pull_on_ubuntu.sh lyq@123.45.67.89
#        bash pull_on_ubuntu.sh myserver        # 如果你在 ~/.ssh/config 里给服务器起了别名
#
# 传输内容(合计约 1.6G,不含 63G 的训练 outputs):
#   - 项目代码(排除 outputs/ 缓存)
#   - 两个冠军权重 s1_track_domrand(s3407 step2000 / s3408 step1500)
#   - hand_data51 评测 val 集(zgz 两条序列,排除用不到的 .evc)+ splits/buckets
# 完成后项目在 ~/lyq/EventHands1/,数据在 ~/lyq/EventHands1/data/hand_data51/
set -eu

if [ $# -lt 1 ]; then
    echo "用法: bash pull_on_ubuntu.sh <服务器SSH地址>  (例: lyq@1.2.3.4 或 config 别名)"
    exit 2
fi
SRV="$1"
SRC="/data1/lyq/code/mesh/EventHands1"
DEST="$HOME/lyq/EventHands1"
CK1="outputs/semkine/s1_track_domrand_s3407/s1_track_domrand_s3407-step=2000.ckpt"
CK2="outputs/semkine/s1_track_domrand_s3408/s1_track_domrand_s3408-step=1500.ckpt"

echo "==== 目标目录: $DEST  ====（会多次提示输入服务器密码）"
mkdir -p "$DEST" "$DEST/data/hand_data51"

echo "==== [1/4] 拉取代码(排除 outputs / 缓存)===="
rsync -aP \
    --exclude 'outputs/' --exclude '.pytest_cache/' --exclude '__pycache__/' \
    --exclude '*.pyc' \
    "$SRV:$SRC/" "$DEST/"

echo "==== [2/4] 拉取两个冠军权重 ===="
mkdir -p "$DEST/outputs/semkine/s1_track_domrand_s3407" \
         "$DEST/outputs/semkine/s1_track_domrand_s3408"
rsync -aP "$SRV:$SRC/$CK1" "$DEST/$CK1"
rsync -aP "$SRV:$SRC/$CK2" "$DEST/$CK2"
rsync -aP "$SRV:$SRC/outputs/semkine/s1_track_domrand_s3407/selection_val_core_step50.json" \
          "$DEST/outputs/semkine/s1_track_domrand_s3407/" 2>/dev/null || true
rsync -aP "$SRV:$SRC/outputs/semkine/s1_track_domrand_s3408/selection_val_core_step50.json" \
          "$DEST/outputs/semkine/s1_track_domrand_s3408/" 2>/dev/null || true

echo "==== [3/4] 拉取 hand_data51 评测 val 集(排除 .evc)===="
rsync -aP --exclude '*.evc' "$SRV:$SRC/data/hand_data51/val" "$DEST/data/hand_data51/"
for f in splits.json splits_semkine.json _retired_splits_semkine_5v2v3.json summary.json sequence_manifest.csv; do
    rsync -aP "$SRV:$SRC/data/hand_data51/$f" "$DEST/data/hand_data51/" 2>/dev/null || true
done
rsync -aP "$SRV:$SRC/data/hand_data51/buckets" "$DEST/data/hand_data51/" 2>/dev/null || true

echo "==== [4/4] 校验关键文件 MD5 ===="
cd "$DEST/data/hand_data51/val" && echo 'eaaedf1809cdd17d90cbe6f26e90e924  zgz_global_events.npy
d7f8fef8fcfd1b9b9e661ed88f701f98  zgz_global.meta
05390fb6f9c941320804b39a5dd83637  zgz_global_tsub.npy
6a9f994bc140d84f454fd67e5aed9490  zgz_local_events.npy
89f776682e8dc859396ef382f2ef2bf9  zgz_local.meta
bad845019c9c7cf334a802955843b711  zgz_local_tsub.npy' | md5sum -c -
cd "$DEST" && echo '49a53b110f6754ee863a71af3b891fc0  '"$CK1"'
cc8f3dadaf8dbd68ed84b5f5e4c5fa21  '"$CK2" | md5sum -c -

echo ""
echo "==== 完成。项目: $DEST ===="
echo "复现评测示例(在 $DEST 下,用你的训练环境 EventHandsTrain):"
echo "  python semkine/eval_track.py --config configs/semkine/s1_track_domrand.yaml \\"
echo "    --ckpt $CK2 --split val_core --step-ms 50"
echo "  预期 zgz val_core RA-MPJPE ≈ 13.6mm(s3408 step1500)/ 14.6mm(s3407 step2000)"
