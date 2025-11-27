#!/usr/bin/bash
#SBATCH -J clip-text-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o logs/slurm-%A.out
#SBATCH -e logs/slurm-%A.err

echo "[INFO] Job started on $(hostname) at $(date)"

# ---------- 1️⃣ Conda 환경 설정 ----------
source /data/hsg0113/anaconda3/etc/profile.d/conda.sh
conda activate momentdetr
echo "[INFO] Conda environment activated: $CONDA_DEFAULT_ENV"

# ---------- 2️⃣ 경로 설정 ----------
REPO_DIR=/data/hsg0113/repos/moment_detr

SRC_TAR=/data/hsg0113/datasets/tarfiles/qvhilights_text.tar.gz
LOCAL_BASE=/tmp/hsg0113_text_${SLURM_JOB_ID}
LOCAL_DATA_DIR=$LOCAL_BASE/data

OUTPUT_DIR=/data/hsg0113/datasets/output/text_features
SCRIPT=$REPO_DIR/extract_text_features.py
MODEL=ViT-B/32

# JSONL 및 SUBS 파일 3종류(split)에 대한 배열 정의
JSONL_LIST=(
    "$REPO_DIR/data/highlight_train_release.jsonl"
    "$REPO_DIR/data/highlight_val_release.jsonl"
    "$REPO_DIR/data/highlight_test_release.jsonl"
)
SUBS_LIST=(
    "$REPO_DIR/data/subs_train.jsonl"
    "$REPO_DIR/data/subs_val.jsonl"
    "$REPO_DIR/data/subs_test.jsonl"
)
SPLIT_NAME=("train" "val" "test")

mkdir -p $LOCAL_DATA_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $REPO_DIR/logs

# ---------- 4️⃣ 3개 split 각각에 대해 텍스트 feature 추출 ----------
cd $REPO_DIR

for i in 0 1 2; do
    CUR_JSONL=${JSONL_LIST[$i]}
    CUR_SUBS=${SUBS_LIST[$i]}
    CUR_SPLIT=${SPLIT_NAME[$i]}

    echo "--------------------------------------------------------------"
    echo "[INFO] Extracting CLIP text features for split: ${CUR_SPLIT}"
    echo "[INFO] JSONL: ${CUR_JSONL}"
    echo "[INFO] SUBS : ${CUR_SUBS}"
    echo "--------------------------------------------------------------"

    python $SCRIPT \
      --jsonl $CUR_JSONL \
      --subs $CUR_SUBS \
      --output_dir $OUTPUT_DIR \
      --model $MODEL

    STATUS=$?
    if [ $STATUS -ne 0 ]; then
      echo "[ERROR] Feature extraction failed for split ${CUR_SPLIT} (exit ${STATUS})"
      exit 1
    fi
done

# ---------- 5️⃣ 완료 후 정리 ----------
echo "[INFO] Cleaning up local scratch..."
rm -rf $LOCAL_BASE
echo "[INFO] Temporary data removed."

echo "[INFO] All text features saved to $OUTPUT_DIR"
echo "[INFO] Job completed successfully at $(date)"
