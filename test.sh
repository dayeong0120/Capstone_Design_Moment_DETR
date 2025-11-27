#!/usr/bin/bash
#SBATCH -J text_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 3:00:00
#SBATCH -o logs/text_test-%A.out

echo "[INFO] Node hostname: $(hostname)"
echo "[INFO] Time: $(date)"
echo "--------------------------------------"

# 1️⃣ 전체 디스크 사용 현황
df -h / /tmp /data /home || echo "[WARN] df command failed"

echo "--------------------------------------"

# 2️⃣ /tmp 내 각 사용자별 용량 (10명까지 표시)
echo "[INFO] Top disk usage in /tmp:"
du -sh /tmp/* 2>/dev/null | sort -hr | head -n 10

echo "--------------------------------------"

# 3️⃣ 현재 사용자별 /tmp 디렉터리 용량
if [ -d "/tmp/$USER" ]; then
    echo "[INFO] Your /tmp/$USER directory size:"
    du -sh /tmp/$USER 2>/dev/null || echo "not found"
else
    echo "[INFO] No personal directory found under /tmp for $USER"
fi

echo "--------------------------------------" echo "[INFO] Done."
