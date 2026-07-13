#!/usr/bin/env bash
# 数据集处理流程：v2 采集数据 → v3 转换 → 过滤异常轨迹 → augment stats → patch
#
# 用法:
#   bash scripts/process_dataset_pipeline.sh \
#       --repo-id  data_toiletButton_0703 \
#       --v2-root  /home/user/.cache/huggingface/lerobot/lerobot_v3/data_toiletButton_0703_f \
#       --v3-root  /media/Data/huggingface/lerobot/lerobot_v3/data_toiletButton_0703_f \
#       --episode-success success     # 可选，不传则交互确认
#       --new-dim 7                   # 可选，action/observation.state 维度（默认：7）
#
# 依赖:
#   conda activate lerobot312
#   工作目录: ~/lerobot-v3/lerobot-v3/src  (lerobot python 包所在位置)
#   evo_rl 脚本目录: /home/user/code_1/evo_rl/scripts

set -e  # 任意一步失败即退出

# ── 参数解析 ─────────────────────────────────────────────────────────────────
REPO_ID=""
V2_ROOT=""
V3_ROOT=""
EPISODE_SUCCESS=""  # 空=交互确认
NEW_DIM=7           # 默认 7 维

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id)          REPO_ID="$2";          shift 2 ;;
        --v2-root)          V2_ROOT="$2";           shift 2 ;;
        --v3-root)          V3_ROOT="$2";           shift 2 ;;
        --episode-success)  EPISODE_SUCCESS="$2";   shift 2 ;;
        --new-dim)          NEW_DIM="$2";           shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 参数检查
if [[ -z "$REPO_ID" || -z "$V2_ROOT" || -z "$V3_ROOT" ]]; then
    echo "用法: $0 --repo-id <id> --v2-root <path> --v3-root <path> [--episode-success success|failure] [--new-dim <dim>]"
    exit 1
fi

LEROBOT_SRC="/home/user/lerobot-v3/lerobot-v3/src"
EVO_RL="/home/user/code_1/evo_rl"

echo "========================================"
echo "数据集处理流程"
echo "  repo-id:          $REPO_ID"
echo "  v2 原始路径:      $V2_ROOT"
echo "  v3 输出路径:      $V3_ROOT"
echo "  episode-success:  ${EPISODE_SUCCESS:-（交互确认）}"
echo "  new-dim:          $NEW_DIM"
echo "========================================"
echo ""

# ── Step 1: v2 → v3 转换 ────────────────────────────────────────────────────
echo "[Step 1/4] v2 → v3 格式转换"
cd "$LEROBOT_SRC"
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id "$REPO_ID" \
    --root "$V2_ROOT" \
    --push-to-hub false
echo "  ✓ 转换完成 → $V2_ROOT"
echo ""

# ── Step 2: 过滤异常长度轨迹 ────────────────────────────────────────────────
echo "[Step 2/4] 过滤异常长度轨迹（IQR 自动检测）"
cd "$EVO_RL"
python delete_ep21.py \
    --root "$V3_ROOT" \
    --auto-filter
echo "  ✓ 异常轨迹过滤完成"
echo ""

# ── Step 3: augment quantile stats ──────────────────────────────────────────
echo "[Step 3/4] augment_dataset_quantile_stats"
cd "$LEROBOT_SRC"
python -m lerobot.datasets.v30.augment_dataset_quantile_stats \
    --repo-id="${REPO_ID}" \
    --root="${V3_ROOT}/" \
    --overwrite
echo "  ✓ stats 更新完成"
echo ""

# ── Step 4: patch for piper_follower ────────────────────────────────────────
echo "[Step 4/4] patch_dataset_for_piper_follower"
cd "$EVO_RL"

if [[ -n "$EPISODE_SUCCESS" ]]; then
    # 命令行已指定，直接传入
    python scripts/patch_dataset_for_piper_follower.py \
        --root "$V3_ROOT" \
        --episode-success "$EPISODE_SUCCESS" \
        --new-dim "$NEW_DIM"
else
    # 交互确认（ask 模式，脚本内部会提示用户输入）
    python scripts/patch_dataset_for_piper_follower.py \
        --root "$V3_ROOT" \
        --episode-success ask \
        --new-dim "$NEW_DIM"
fi
echo "  ✓ patch 完成"
echo ""

echo "========================================"
echo "全部完成！数据集路径: $V3_ROOT"
echo "========================================"
