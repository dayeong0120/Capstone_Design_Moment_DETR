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
# CLIP Text Feature Extraction (GPU, /tmp SSD workflow)
# ----------------------------------------------------------
# Conda env : momentdetr
# Source archive : /data/hsg0113/datasets/tarfiles/qvhilights_text.tar.gz
# Local scratch : /tmp/hsg0113_text_${SLURM_JOB_ID}/
# Output backup : /data/hsg0113/datasets/output/text_features/
# ==========================================================

echo "[INFO] Job started on $(hostname) at $(date)"

# ---------- 1️⃣ Conda 환경 설정 ----------
source /data/hsg0113/anaconda3/etc/profile.d/conda.sh
conda activate momentdetr
echo "[INFO] Conda environment activated: $CONDA_DEFAULT_ENV"

# ---------- 2️⃣ 경로 설정 ----------
REPO_DIR=/data/hsg0113/repos/moment_detr
SRC_TAR=/data/hsg0113/datasets/tarfiles/qvhilights_text.tar.gz   # NAS에 있는 원본 압축 파일
LOCAL_BASE=/tmp/hsg0113_text_${SLURM_JOB_ID}
LOCAL_DATA_DIR=$LOCAL_BASE/data
OUTPUT_DIR=/data/hsg0113/datasets/output/text_features
SCRIPT=$REPO_DIR/extract_text_features.py
MODEL=ViT-B/32

JSONL=$REPO_DIR/data/highlight_train_release.jsonl
SUBS=$REPO_DIR/data/subs_train.jsonl

mkdir -p $LOCAL_DATA_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $REPO_DIR/logs

# ---------- 3️⃣ 데이터 복사 및 압축 해제 ----------
echo "[INFO] Copying dataset archive to local /tmp..."
cp $SRC_TAR $LOCAL_BASE/ || { echo "[ERROR] Failed to copy dataset"; exit 1; }

echo "[INFO] Extracting dataset..."
tar -xf $LOCAL_BASE/qvhilights_text.tar.gz -C $LOCAL_DATA_DIR || { echo "[ERROR] Failed to extract dataset"; exit 1; }
echo "[INFO] Extraction complete: $(du -sh $LOCAL_DATA_DIR)"

# ---------- 4️⃣ 실행 ----------
echo "[INFO] Starting CLIP text feature extraction..."
cd $REPO_DIR

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

# ---------- 5️⃣ 완료 후 정리 ----------
echo "[INFO] Cleaning up local scratch..."
rm -rf $LOCAL_BASE
echo "[INFO] Temporary data removed."

echo "[INFO] Text features saved to $OUTPUT_DIR"
echo "[INFO] Job completed successfully at $(date)"
