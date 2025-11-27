dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=/data/hsg0113/repos/moment_detr/results
exp_id=exp

######## data paths
train_path=/data/hsg0113/repos/moment_detr/data/highlight_train_release.jsonl
eval_path=/data/hsg0113/repos/moment_detr/data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
SLURM_JOB_ID=${SLURM_JOB_ID:-manual_$(date +%H%M%S)}
feat_root=/tmp/hsg0113_features_${SLURM_JOB_ID}/features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# ---------------- TEXT FEATURES ---------------- #
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/text_features/
  t_feat_dim=512
else
  echo "[ERROR] Wrong arg for t_feat_type: ${t_feat_type}"
  exit 1
fi

#### training
bsz=32

# ----------- 실행 로그 ----------- #
echo "[INFO] Training script started"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[INFO] train_path=${train_path}"
echo "[INFO] eval_path=${eval_path}"
echo "[INFO] feat_root=${feat_root}"
echo "[INFO] v_feat_dirs=${v_feat_dirs[@]}"
echo "[INFO] t_feat_dir=${t_feat_dir}"

echo "[DEBUG] Checking Python entrypoint..."
python -c "print('>>> Running moment_detr/train.py manual test OK')"

echo "[INFO] Starting Moment-DETR training..."
PYTHONPATH=$PYTHONPATH:/data/hsg0113/repos/moment_detr python - <<EOF
import sys
sys.argv = [
  "train.py",
  "--dset_name", "hl",
  "--ctx_mode", "video_tef",
  "--train_path", "/data/hsg0113/repos/moment_detr/data/highlight_train_release.jsonl",
  "--eval_path", "/data/hsg0113/repos/moment_detr/data/highlight_val_release.jsonl",
  "--eval_split_name", "val",
  "--v_feat_dirs", "/tmp/hsg0113_features_${SLURM_JOB_ID}/features/slowfast_features", "/tmp/hsg0113_features_${SLURM_JOB_ID}/features/clip_features",
  "--v_feat_dim", "2816",
  "--t_feat_dir", "${t_feat_dir}",          # ✅ train & eval 동일
  "--t_feat_dim", "512",
  "--bsz", "32",
  "--results_root", "/data/hsg0113/repos/moment_detr/results",
  "--exp_id", "exp_qvh_features_${SLURM_JOB_ID}"
]
from moment_detr.train import start_training
print("DEBUG sys.argv =", sys.argv)
# start_training()
EOF

# For Debugging
export PYTHONUNBUFFERED=1
