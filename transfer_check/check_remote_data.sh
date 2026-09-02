#!/usr/bin/env bash
# 在目标 ubuntu 机器上运行:检查本机数据与训练机(192.168.50.26)是否对齐。
# 用法: bash check_remote_data.sh [数据目录]
#   例:  bash check_remote_data.sh /home/ubuntu/lyq/hand_data
#   不带参数时自动在常见路径下搜索。
# 能识别两种数据:
#   A. 加工后的 hand_data51(评测直接可用)—— 逐文件 MD5 校验
#   B. 原始采集 hand_data(AEDAT4+mesh 等)—— 按 zgz 关键文件大小比对
# 基准生成于 2026-08-31,对应 15mm 评测(s1_track_domrand @ zgz val)。
set -u

# ---- 基准 A1:zgz 评测序列文件(hand_data51/val/,按文件名校验) ----
VAL_MD5='6cb50a263ee513182d7c0b444da99dbc  zgz_global_aux.npz
2a2edbf76f40e3eb729ed7a770331e52  zgz_global.done.json
eaaedf1809cdd17d90cbe6f26e90e924  zgz_global_events.npy
d7f8fef8fcfd1b9b9e661ed88f701f98  zgz_global.meta
c8e396fc52992dce89be2fbe36baddf3  zgz_global_offsets.npy
05390fb6f9c941320804b39a5dd83637  zgz_global_tsub.npy
1ef4495fa738d5038ca5867576a58e83  zgz_local_aux.npz
69b6858b63aa3574a38987a03f9058b9  zgz_local.done.json
6a9f994bc140d84f454fd67e5aed9490  zgz_local_events.npy
89f776682e8dc859396ef382f2ef2bf9  zgz_local.meta
577878206a7739c5442c1e887d43aa65  zgz_local_offsets.npy
bad845019c9c7cf334a802955843b711  zgz_local_tsub.npy'

# ---- 基准 A2:数据集根目录文件(hand_data51/) ----
ROOT_MD5='62fe8552c752cb4b640e0c892137f4a3  splits.json
a2bd166584df625d810db6cb7c1a8891  splits_semkine.json
d891d8e8c3f6912df337b8d36e2e5c05  _retired_splits_semkine_5v2v3.json
d954e823eb8aaf07805d1be7360d359b  summary.json
36f0b69ff4ef7e31364792f118510ce1  sequence_manifest.csv
19149c66f9759f7e0300f0940b98f4e2  buckets/summary_step50.json
4d17e9044943f3ad27998f1588c68582  buckets/test_step50.json
e88f4c22979ec66b73b61c32ce105e87  buckets/train_step50.json
57a6a6a98f28ad77a7315e5dab9ddb5b  buckets/val_step50.json'

# ---- 基准 B:原始数据 zgz 关键文件的字节数(路径相对序列目录) ----
RAW_SIZES='233207455 zgz_global/event/zgz_global_events.aedat4
72932688 zgz_global/mesh/zgz_global_mesh.npz
14381264 zgz_local/event/zgz_local_events.aedat4
68895192 zgz_local/mesh/zgz_local_mesh.npz'

echo "==== 主机: $(hostname)  用户: $(whoami)  时间: $(date '+%F %T') ===="

if [ $# -ge 1 ]; then
    SEARCH_ROOTS="$1"
else
    SEARCH_ROOTS="$HOME /home/ubuntu/lyq /data /data1 /data2 /mnt /media /srv /opt /home"
fi

NPY_HITS=""
AEDAT_HITS=""
DIR_HITS=""
for r in $SEARCH_ROOTS; do
    [ -d "$r" ] || continue
    NPY_HITS="$NPY_HITS
$(find "$r" -maxdepth 8 -name zgz_global_events.npy 2>/dev/null)"
    AEDAT_HITS="$AEDAT_HITS
$(find "$r" -maxdepth 8 -name zgz_global_events.aedat4 2>/dev/null)"
    DIR_HITS="$DIR_HITS
$(find "$r" -maxdepth 8 -type d \( -name 'hand_data*' -o -name 'zgz_global' \) 2>/dev/null)"
done
NPY_HITS=$(echo "$NPY_HITS" | grep -v '^$' | sort -u || true)
AEDAT_HITS=$(echo "$AEDAT_HITS" | grep -v '^$' | sort -u || true)
DIR_HITS=$(echo "$DIR_HITS" | grep -v '^$' | sort -u || true)

echo ""
echo "== [1/3] 数据目录搜索结果 =="
echo "-- hand_data* / zgz_global 目录:"
echo "${DIR_HITS:-  (无)}"
echo "-- 加工数据标志 zgz_global_events.npy:"
echo "${NPY_HITS:-  (无)}"
echo "-- 原始数据标志 zgz_global_events.aedat4:"
echo "${AEDAT_HITS:-  (无)}"

FOUND_ANY=0

if [ -n "${NPY_HITS:-}" ]; then
    FOUND_ANY=1
    echo "$NPY_HITS" | while IFS= read -r f; do
        d=$(dirname "$f")
        echo ""
        echo "==== [2/3] 加工数据(hand_data51)MD5 校验: $d ===="
        ( cd "$d" && echo "$VAL_MD5" | md5sum -c - 2>&1 )
        p=$(dirname "$d")
        if [ -f "$p/splits.json" ] || [ -f "$p/splits_semkine.json" ]; then
            echo "-- 根目录 $p:"
            ( cd "$p" && echo "$ROOT_MD5" | md5sum -c - 2>&1 )
        else
            echo "-- (上级目录 $p 没有 splits*.json,结构与训练机不同)"
        fi
    done
else
    echo ""
    echo "==== [2/3] 未找到加工数据(hand_data51)。评测所需的 .npy/.meta/_tsub 不在本机。 ===="
fi

if [ -n "${AEDAT_HITS:-}" ]; then
    FOUND_ANY=1
    echo ""
    echo "==== [3/3] 原始数据(AEDAT4)大小比对 ===="
    echo "$AEDAT_HITS" | while IFS= read -r f; do
        # f = .../zgz_global/event/zgz_global_events.aedat4 -> base = zgz_global 的上级
        base=$(dirname "$(dirname "$(dirname "$f")")")
        echo "-- 序列根目录: $base"
        echo "$RAW_SIZES" | while read -r want rel; do
            g="$base/$rel"
            if [ -f "$g" ]; then
                got=$(wc -c < "$g" | tr -d ' ')
                if [ "$got" = "$want" ]; then st="OK"; else st="大小不同(期望 $want)"; fi
                echo "   $rel : $got 字节 -> $st"
            else
                echo "   $rel : 缺失"
            fi
        done
        ls "$base" 2>/dev/null | head -5 | sed 's/^/   同级内容: /'
    done
else
    echo ""
    echo "==== [3/3] 未找到原始数据(AEDAT4)。 ===="
fi

if [ "$FOUND_ANY" = "0" ]; then
    echo ""
    echo "结论: 两种数据都没找到。"
fi

echo ""
echo "==== 完成。请把以上全部输出粘贴回聊天。 ===="
