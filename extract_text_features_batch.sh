#!/usr/bin/bash
#SBATCH -J clip-text-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o logs/slurm-%A.out
#SBATCH -e logs/slurm-%A.err

# ==========================================================
# CLIP Text Feature Extraction (GPU)
# ----------------------------------------------------------
# Conda env : momentdetr
# Input : highlight_train_release.jsonl + subs_train.jsonl
# Output : /data/hsg0113/datasets/output/text_features
# Script : extract_text_features_final.py
# ==========================================================

echo "[INFO] Job started on $(hostname) at $(date)"

# ---------- 1️⃣ Conda 환경 설정 ----------
source /data/hsg0113/anaconda3/etc/profile.d/conda.sh
conda activate momentdetr
echo "[INFO] Conda environment activated: $CONDA_DEFAULT_ENV"

# ---------- 2️⃣ 경로 설정 ----------
REPO_DIR=/data/hsg0113/repos/moment_detr
JSONL=$REPO_DIR/data/highlight_train_release.jsonl
SUBS=$REPO_DIR/data/subs_train.jsonl
OUTPUT_DIR=/data/hsg0113/datasets/output/text_features
SCRIPT=$REPO_DIR/extract_text_features.py
MODEL=ViT-B/32

mkdir -p $OUTPUT_DIR
mkdir -p $REPO_DIR/logs

echo "[INFO] Input JSONL: $JSONL"
echo "[INFO] Subs file  : $SUBS"
echo "[INFO] Output Dir : $OUTPUT_DIR"
echo "[INFO] Using model: $MODEL"

# ---------- 3️⃣ 실행 ----------
echo "[INFO] Starting text feature extraction..."
python $SCRIPT \
  --jsonl $JSONL \
  --subs $SUBS \
  --output_dir $OUTPUT_DIR \
  --model $MODEL

STATUS=$?
if [ $STATUS -ne 0 ]; then
  echo "[ERROR] Feature extraction failed with exit code $STATUS"
  exit 1
fi

# ---------- 4️⃣ 완료 처리 ----------
echo "[INFO] Text features saved to $OUTPUT_DIR"
echo "[INFO] Job completed successfully at $(date)"
