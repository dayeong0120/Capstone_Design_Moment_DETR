#!/usr/bin/bash
#SBATCH -J analyze_momentdetr             # Job 이름
#SBATCH --gres=gpu:0                      # GPU 코어 수
#SBATCH --cpus-per-task=8                 # CPU 코어 수
#SBATCH --mem=32G                         # 메모리
#SBATCH -p batch_ce_ugrad                 # 사용할 파티션 이름
#SBATCH -w moana-y2                       # 사용할 GPU 노드
#SBATCH -t 0-04:00:00                     # 최대 실행 시간 (4시간)
#SBATCH -o logs/slurm-%A.out              # 표준 출력 로그
#SBATCH -e logs/slurm-%A.err              # 에러 로그

# ==========================================================
# Moment-DETR Inference Error Analysis
# Author: 한승규
# ==========================================================

echo "[INFO] Job started on $(hostname) at $(date)"

# ---------- 1️⃣ Conda 환경 활성화 ----------
source /data/hsg0113/anaconda3/etc/profile.d/conda.sh
conda activate momentdetr
echo "[INFO] Conda environment activated: $CONDA_DEFAULT_ENV"

# ---------- 2️⃣ 경로 설정 ----------
cd /data/hsg0113/repos/moment_detr/viz
mkdir -p logs

# ---------- 3️⃣ 분석 실행 ----------
echo "[INFO] Running analysis in $(pwd)..."
python analyze_moment_detr_errors.py \
    --pred_path inference_hl_val_None_preds.jsonl \
    --gt_path highlight_val_release.jsonl \
    --output_path worst_100_preds.jsonl

# ---------- 4️⃣ 결과 요약 ----------
echo "[INFO] Analysis completed at $(date)"
echo "[INFO] Generated files:"
ls -lh worst_100_preds.jsonl iou_hist.png 2>/dev/null || echo "[WARN] Output files not found!"

