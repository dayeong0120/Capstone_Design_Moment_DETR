#!/usr/bin/bash
#SBATCH -J capdetr_train_tmp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -w moana-y2
#SBATCH -t 1-00:00:00
#SBATCH -o logs/slurm-%A.out
#SBATCH -e logs/slurm-%A.err

# ==========================================================
# Moment-DETR Training with /tmp SSD scratch
# ----------------------------------------------------------
# Conda env : momentdetr
# Input archive : /data/hsg0113/datasets/output/features_all.tar.gz
# Local scratch : /tmp/hsg0113_features_${SLURM_JOB_ID}/
# Output backup : /data/hsg0113/datasets/training_results/
# ==========================================================

echo "[INFO] Job started on $(hostname) at $(date)"

# ---------- 1️⃣ Conda 환경 설정 ----------
source /data/hsg0113/anaconda3/etc/profile.d/conda.sh
conda activate momentdetr
echo "[INFO] Conda environment activated: $CONDA_DEFAULT_ENV"

# ---------- 2️⃣ 경로 설정 ----------
REPO_DIR=/data/hsg0113/repos/cap_detr
SRC_TAR=/data/hsg0113/datasets/output/features_all.tar.gz
LOCAL_BASE=/tmp/hsg0113_features_${SLURM_JOB_ID}
LOCAL_DATA_DIR=$LOCAL_BASE
TRAIN_JSON=$REPO_DIR/data/highlight_train_release.jsonl
VAL_JSON=$REPO_DIR/data/highlight_val_release.jsonl
BACKUP_DIR=/data/hsg0113/datasets/training_results/exp_qvh_features_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M)

mkdir -p $LOCAL_DATA_DIR
mkdir -p $BACKUP_DIR

# ---------- 3️⃣ 데이터 복사 및 압축 해제 ----------
echo "[INFO] Copying and extracting dataset to local /tmp..."
cp $SRC_TAR $LOCAL_BASE/ || { echo "[ERROR] Failed to copy dataset"; exit 1; }
tar -xf $LOCAL_BASE/features_all.tar.gz -C $LOCAL_DATA_DIR || { echo "[ERROR] Failed to extract dataset"; exit 1; }
echo "[INFO] Data extracted to $LOCAL_DATA_DIR"

# ---------- 4️⃣ 디스크 용량 확인 ----------
DISK_AVAIL=$(df -h /tmp | tail -1 | awk '{print $4}')
echo "[INFO] Available space on /tmp: $DISK_AVAIL"
du -sh $LOCAL_DATA_DIR || echo "[WARN] Unable to calculate dataset size."

# ---------- 5️⃣ 출력 폴더 준비 ----------
cd $REPO_DIR
mkdir -p logs tensorboard ckpt

# ---------- 6️⃣ 학습 시작 ----------
echo "[INFO] Starting Cap-DETR training..."
bash scripts/train.sh \
  --exp_id exp_qvh_features_${SLURM_JOB_ID} \

# ---------- 8️⃣ 결과 백업 ----------
echo "[INFO] Backing up logs and checkpoints to NAS..."
rsync -avh --progress $REPO_DIR/logs $REPO_DIR/tensorboard $REPO_DIR/ckpt $BACKUP_DIR/ \
    || echo "[WARN] Backup incomplete, check NAS access."
echo "[INFO] Backup saved to $BACKUP_DIR"

# ---------- 9️⃣ 임시 데이터 정리 ----------
echo "[INFO] Cleaning up /tmp scratch..."
rm -rf $LOCAL_BASE
echo "[INFO] Temporary data removed."

echo "[INFO] Job completed at $(date)"
