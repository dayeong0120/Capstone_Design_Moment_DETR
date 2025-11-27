#!/usr/bin/bash
#SBATCH -J momentdetr_train_tmp
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
REPO_DIR=/data/hsg0113/repos/moment_detr
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

# 🔧 이름 맞추기
mv $LOCAL_DATA_DIR/clip-vit_features $LOCAL_DATA_DIR/clip_features
mv $LOCAL_DATA_DIR/text_features $LOCAL_DATA_DIR/clip_text_features
echo "[INFO] Renamed feature folders for train.sh compatibility."

# ✅ 복사 검증 로그 추가
echo "[INFO] Listing copied clip_text_features directory contents (first 10 files):"
ls -lh $LOCAL_DATA_DIR/clip_text_features | head -n 10 || echo "[WARN] clip_text_features folder not found!"
echo "[INFO] Total .npz files: $(ls $LOCAL_DATA_DIR/clip_text_features/*.npz 2>/dev/null | wc -l)"

# ✅ Feature 매칭 검증
echo "[DEBUG] Checking JSON↔feature file match..."
python - <<EOF
import os, json
json_path = "/data/hsg0113/repos/moment_detr/data/highlight_train_release.jsonl"
v_feat_dir = "$LOCAL_DATA_DIR/clip_features"
t_feat_dir = "$LOCAL_DATA_DIR/clip_text_features"
print("Checking paths:")
print("v_feat_dir =", v_feat_dir)
print("t_feat_dir =", t_feat_dir)
with open(json_path) as f:
    data = [json.loads(x) for x in f]
print(f"Total JSON samples: {len(data)}")
cnt_v, cnt_t = 0, 0
for d in data[:10]:
    qid = d["qid"]
    vid = d["vid"]
    v_ok = os.path.exists(os.path.join(v_feat_dir, f"{vid}.npz"))
    t_ok = os.path.exists(os.path.join(t_feat_dir, f"qid{qid}.npz"))
    print(f"VID {vid}: {v_ok} | QID {qid}: {t_ok}")
    if v_ok: cnt_v += 1
    if t_ok: cnt_t += 1
print(f"[SUMMARY] first 10 samples — video OK: {cnt_v}, text OK: {cnt_t}")
EOF


# ---------- 4️⃣ 디스크 용량 확인 ----------
DISK_AVAIL=$(df -h /tmp | tail -1 | awk '{print $4}')
echo "[INFO] Available space on /tmp: $DISK_AVAIL"
du -sh $LOCAL_DATA_DIR || echo "[WARN] Unable to calculate dataset size."

# ---------- 5️⃣ 출력 폴더 준비 ----------
cd $REPO_DIR
mkdir -p logs tensorboard ckpt

# ---------- 6️⃣ 학습 시작 ----------
echo "[INFO] Starting Moment-DETR training..."
bash scripts/train.sh \
  --exp_id exp_qvh_features_${SLURM_JOB_ID} \

# ---------- 7️⃣ 평가 ----------
# BEST_CKPT=$REPO_DIR/ckpt/exp_qvh_features_${SLURM_JOB_ID}_best.ckpt
# if [ -f "$BEST_CKPT" ]; then
#     echo "[INFO] Training finished. Starting evaluation..."
#     python -m moment_detr.inference \
#         --resume "$BEST_CKPT" \
#         --eval_path $VAL_JSON \
#         --eval_split_name val
# else
#     echo "[WARN] No best checkpoint found. Skipping evaluation."
# fi

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
