#!/usr/bin/bash
#SBATCH -J analyze_momentdetr_fg_bg       # Job 이름
#SBATCH --gres=gpu:0                      # GPU 사용 안함
#SBATCH --cpus-per-task=8                 # CPU 코어 수
#SBATCH --mem=32G                         # 메모리
#SBATCH -p batch_ce_ugrad                 # 파티션
#SBATCH -w moana-y2                       # 노드
#SBATCH -t 0-04:00:00                     # 최대 실행 시간
#SBATCH -o logs/slurm-%A.out              # 표준 출력 로그
#SBATCH -e logs/slurm-%A.err              # 에러 로그

# ==========================================================
# Moment-DETR Foreground / Background Error Analysis
# ==========================================================

echo "[INFO] Job started on $(hostname) at $(date)"

# ---------- 1️⃣ Conda 환경 활성화 ----------
source /data/hsg0113/anaconda3/etc/profile.d/conda.sh
conda activate momentdetr
echo "[INFO] Conda environment activated: $CONDA_DEFAULT_ENV"

# ---------- 2️⃣ 경로 설정 ----------
cd /data/hsg0113/repos/moment_detr/viz
mkdir -p logs
echo "[INFO] Working directory: $(pwd)"

# ---------- 3️⃣ 출력 폴더 자동 생성 ----------
OUTPUT_DIR="analysis_fg_bg3"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "[INFO] Created output directory: $OUTPUT_DIR"
else
    echo "[INFO] Output directory already exists: $OUTPUT_DIR"
fi

# ---------- 4️⃣ 분석 실행 ----------
echo "[INFO] Running detailed foreground/background analysis..."
python analyze_moment_detr_errors_fg_bg.py \
    --pred_path inference_hl_val_None_preds.jsonl \
    --gt_path highlight_val_release.jsonl \
    --conf_thresh 0.5 \
    --worst_k 50 \
    --output_dir "$OUTPUT_DIR"

STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "[ERROR] Python script failed with exit code $STATUS"
    exit $STATUS
fi

# ---------- 5️⃣ 결과 요약 ----------
echo "[INFO] Analysis completed at $(date)"
echo "[INFO] Generated outputs in $OUTPUT_DIR/:"
ls -lh "$OUTPUT_DIR"/* 2>/dev/null || echo "[WARN] No output files found!"

# ---------- 6️⃣ 통계 요약 ----------
if [ -f "$OUTPUT_DIR/stats.json" ]; then
    echo "[INFO] ===== Summary Statistics ====="
    cat "$OUTPUT_DIR/stats.json"
else
    echo "[WARN] stats.json not found!"
fi

echo "[INFO] Job finished successfully at $(date)"
