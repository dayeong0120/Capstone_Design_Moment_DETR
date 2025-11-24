# -*- coding: utf-8 -*-
"""
analyze_moment_detr_errors_fg_bg.py

Moment-DETR/Temporal Grounding 결과를 '분석 목적'에 맞게 재가공해
- (1) Foreground로 판단한 예측 중 IoU가 낮은 worst 케이스
- (2) Background로 판단했지만 GT와 IoU가 높은(즉, 잡아냈지만 버린) missed 케이스
를 자동 추출하고,
- (3) 카테고리별 IoU 히스토그램을 저장한다.

※ 본 스크립트는 '평가(metric 산출)'용이 아니라 '원인 분석'을 돕는 용도임.
   그러므로 foreground/background를 나누어 살피고, threshold를 바꿔가며
   분류/탐색 실패를 구분하는 것이 핵심이다.

입출력 형식(가정)
-----------------
- 예측 파일(pred_path): JSON Lines (.jsonl)
  각 줄은 하나의 QID(질의)에 대한 결과를 담고 있고, 다음 필드를 가진다고 가정한다.
    {
      "qid": "Q000123",
      "vid": "V000045",
      "query": "a person opens the fridge",
      "pred_relevant_windows": [
         [start, end, conf],    # 시간 단위(초), conf는 foreground 확률(softmax[...,0]) 등
         ...
      ]
    }

- 정답 파일(gt_path): JSON Lines (.jsonl)
  각 줄은 해당 qid에 대한 GT를 담고 있고, 다음 필드를 가진다고 가정한다.
    {
      "qid": "Q000123",
      "relevant_windows": [
        [gt_start, gt_end],
        ...
      ]
    }

출력
----
output_dir/
 ├─ all_queries.csv                 # 전체 query-level 레코드 (분석의 기본 테이블)
 ├─ worst_foreground.csv            # conf>θ 이지만 IoU<0.3 인 worst FG 케이스 Top N
 ├─ missed_background.csv           # conf≤θ 이면서 IoU≥0.5 인 BG 중 GT 근접 케이스 Top N
 ├─ histogram_iou_fg.png            # Foreground IoU 히스토그램 (단일 플롯)
 ├─ histogram_iou_bg.png            # Background IoU 히스토그램 (단일 플롯)
 └─ stats.json                      # 요약 통계(평균, 중앙값 등)

사용 예
------
python analyze_moment_detr_errors_fg_bg.py \
  --pred_path inference_val.jsonl \
  --gt_path gt_val.jsonl \
  --conf_thresh 0.3 \
  --worst_k 50 \
  --output_dir analysis_fg_bg
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 서버/노트북 환경에서 GUI 없이 저장만 하도록 설정
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# (1) 기초 유틸: Temporal IoU 계산
# ---------------------------------------------------------------------------
def temporal_iou(span1, span2):
    """
    두 시간 구간(span1=[s1,e1], span2=[s2,e2]) 사이의 IoU(교집합/합집합)를 계산한다.
    - 입력:  (float s1, float e1), (float s2, float e2)  (단, e > s 보장 필요)
    - 출력:  0.0 ~ 1.0
    """
    s1, e1 = float(span1[0]), float(span1[1])
    s2, e2 = float(span2[0]), float(span2[1])
    # 교집합 길이: 겹치는 구간의 길이. 음수가 되지 않도록 max(0, ...) 처리
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    # 합집합 길이: 개별 길이 합에서 교집합 길이를 뺀 값
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# (2) 데이터 로더: JSONL을 리스트로 읽기
# ---------------------------------------------------------------------------
def load_jsonl(path):
    """
    JSON Lines(.jsonl) 파일을 리스트(dict 리스트)로 로드한다.
    - 각 줄을 json.loads로 파싱
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# (3) 메인 분석 루틴
# ---------------------------------------------------------------------------
def analyze(pred_path,
            gt_path,
            output_dir="analysis_fg_bg",
            conf_thresh=0.3,
            worst_k=50,
            fg_iou_bad=0.3,
            bg_iou_good=0.5):
    """
    분석 핵심 함수.

    파라미터
    --------
    pred_path : str
        예측 결과(.jsonl) 경로
    gt_path : str
        GT(.jsonl) 경로
    output_dir : str
        결과물 저장 폴더
    conf_thresh : float
        foreground / background를 가르는 confidence 임계값 θ
        - conf > θ : foreground로 간주
        - conf ≤ θ : background로 간주
    worst_k : int
        Top-K 뽑는 수 (worst foreground, missed background 각각에 적용)
    fg_iou_bad : float
        foreground 실패 케이스로 간주할 IoU 상한 (예: 0.3 미만이면 '나쁨')
    bg_iou_good : float
        background인데도 GT 근처로 간주할 IoU 하한 (예: 0.5 이상이면 '좋음')

    동작
    ----
    1) pred_jsonl의 각 레코드(=qid 단위)에 대해, 모든 예측 구간의 (start, end, conf)를 순회
    2) 해당 qid의 GT 구간들과 개별 IoU를 계산하여 "최대 IoU"를 채택
    3) conf와 IoU를 기준으로 fg/bg 레이블 및 카테고리 부여
       - fg_confident_wrong: conf>θ & IoU<fg_iou_bad
       - bg_missed_hit     : conf≤θ & IoU≥bg_iou_good
    4) 전체 레코드를 CSV로 저장하고, 각 카테고리별 Top-K만 별도 CSV로 저장
    5) Foreground/Background 각각 IoU 히스토그램을 저장(단일 플롯 2개)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading predictions: {pred_path}")
    preds = load_jsonl(pred_path)
    print(f"[INFO] Loading ground truths: {gt_path}")
    gt_list = load_jsonl(gt_path)
    gts = {d["qid"]: d for d in gt_list}  # qid → GT dict

    # -----------------------------
    # (3-1) 전체 query-level 레코드 생성
    # -----------------------------
    rows = []
    total_pred_spans = 0

    for p in preds:
        qid = p.get("qid")
        if qid is None or qid not in gts:
            # GT가 없으면 분석 불가 → 스킵
            continue

        vid = p.get("vid")
        query = p.get("query", "")

        # 예측 구간들: [start, end, conf] 형식 가정
        pred_windows = p.get("pred_relevant_windows", [])
        # GT 구간들: [gt_start, gt_end] 형식 가정
        gt_windows = gts[qid].get("relevant_windows", [])

        if not pred_windows or not gt_windows:
            # 예측 또는 GT가 비었으면 IoU 계산 의미 없음 → 스킵
            continue

        # 각 예측 구간을 순회하며, 해당 qid의 GT들과 최대 IoU 계산
        for win in pred_windows:
            # 예측 구간의 포맷 안정성 체크: 길이가 3 미만이면 스킵
            if not isinstance(win, (list, tuple)) or len(win) < 2:
                continue
            s = float(win[0])
            e = float(win[1])
            conf = float(win[2]) if len(win) >= 3 and win[2] is not None else 0.0

            # 잘못된 구간(e <= s)은 스킵 (학습/전처리 오류 방지)
            if not (math.isfinite(s) and math.isfinite(e) and e > s):
                continue

            # GT 여러 개인 경우, 해당 예측 구간과의 IoU 중 '최대값' 채택
            best_iou = 0.0
            for gts_, gte_ in gt_windows:
                iou = temporal_iou((s, e), (float(gts_), float(gte_)))
                if iou > best_iou:
                    best_iou = iou

            cat = "fg" if conf > conf_thresh else "bg"

            rows.append({
                "qid": qid,
                "vid": vid,
                "query": query,
                "pred_start": s,
                "pred_end": e,
                "conf": conf,
                "iou": best_iou,
                "category": cat
            })
            total_pred_spans += 1

    if not rows:
        print("[WARN] No valid prediction/GT pairs found. Nothing to save.")
        return

    df = pd.DataFrame(rows)
    all_csv = output_dir / "all_queries.csv"
    df.to_csv(all_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved all query-level records: {all_csv} ({len(df)} rows, {total_pred_spans} spans)")

    # -----------------------------
    # (3-2) 규칙 기반 카테고리 추가 (분석용)
    # -----------------------------
    df["category_detail"] = "none"
    df.loc[(df["category"] == "fg") & (df["iou"] < fg_iou_bad), "category_detail"] = "fg_confident_wrong"
    df.loc[(df["category"] == "bg") & (df["iou"] >= bg_iou_good), "category_detail"] = "bg_missed_hit"

    # Top-K 추출: 정렬 기준은 분석 목적에 맞게 설정
    # - worst_foreground: IoU 오름차순(낮은 것 우선)
    # - missed_background: IoU 내림차순(높은 것 우선, '아깝게 놓친' 순)
    worst_fg = (
        df[df["category_detail"] == "fg_confident_wrong"]
        .sort_values(["iou", "conf"], ascending=[True, False])
        .head(worst_k)
    )
    missed_bg = (
        df[df["category_detail"] == "bg_missed_hit"]
        .sort_values(["iou", "conf"], ascending=[False, True])
        .head(worst_k)
    )

    worst_fg_csv = output_dir / "worst_foreground.csv"
    missed_bg_csv = output_dir / "missed_background.csv"
    worst_fg.to_csv(worst_fg_csv, index=False, encoding="utf-8-sig")
    missed_bg.to_csv(missed_bg_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved worst_fg({len(worst_fg)}): {worst_fg_csv}")
    print(f"[INFO] Saved missed_bg({len(missed_bg)}): {missed_bg_csv}")

    # -----------------------------
    # (3-3) 히스토그램 저장 (규칙: 한 플롯에 한 차트, 색상 지정하지 않음)
    # -----------------------------
    # Foreground IoU 분포
    fg_ious = df[df["category"] == "fg"]["iou"].values
    if fg_ious.size > 0:
        plt.figure(figsize=(7, 4))
        plt.hist(fg_ious, bins=20)  # 색상 지정 금지 (기본값 사용)
        # 평균/중앙값 가이드라인(색상 미지정)
        plt.axvline(fg_ious.mean(), linestyle="--", linewidth=1, label=f"mean={fg_ious.mean():.3f}")
        plt.axvline(pd.Series(fg_ious).median(), linestyle=":", linewidth=1, label=f"median={pd.Series(fg_ious).median():.3f}")
        plt.title("Foreground IoU Histogram")
        plt.xlabel("IoU")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        fg_png = output_dir / "histogram_iou_fg.png"
        plt.savefig(fg_png)
        plt.close()

    # Background IoU 분포
    bg_ious = df[df["category"] == "bg"]["iou"].values
    if bg_ious.size > 0:
        plt.figure(figsize=(7, 4))
        plt.hist(bg_ious, bins=20)  # 색상 지정 금지 (기본값 사용)
        # 평균/중앙값 가이드라인(색상 미지정)
        plt.axvline(bg_ious.mean(), linestyle="--", linewidth=1, label=f"mean={bg_ious.mean():.3f}")
        plt.axvline(pd.Series(bg_ious).median(), linestyle=":", linewidth=1, label=f"median={pd.Series(bg_ious).median():.3f}")
        plt.title("Background IoU Histogram")
        plt.xlabel("IoU")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        bg_png = output_dir / "histogram_iou_bg.png"
        plt.savefig(bg_png)
        plt.close()

    # -----------------------------
    # (3-4) 간단 통계 요약 저장 (json)
    # -----------------------------
    stats = {
        "total_rows": int(len(df)),
        "conf_thresh": float(conf_thresh),
        "fg_iou_bad": float(fg_iou_bad),
        "bg_iou_good": float(bg_iou_good),
        "counts": {
            "fg": int((df["category"] == "fg").sum()),
            "bg": int((df["category"] == "bg").sum()),
            "fg_confident_wrong": int((df["category"] == "fg_confident_wrong").sum()) if "fg_confident_wrong" in df["category_detail"].values else int((df["category_detail"] == "fg_confident_wrong").sum()),
            "bg_missed_hit": int((df["category_detail"] == "bg_missed_hit").sum()),
        },
        "means": {
            "iou_all": float(df["iou"].mean()),
            "iou_fg": float(df[df["category"] == "fg"]["iou"].mean()) if (df["category"] == "fg").any() else None,
            "iou_bg": float(df[df["category"] == "bg"]["iou"].mean()) if (df["category"] == "bg").any() else None,
        },
        "medians": {
            "iou_all": float(df["iou"].median()),
            "iou_fg": float(df[df["category"] == "fg"]["iou"].median()) if (df["category"] == "fg").any() else None,
            "iou_bg": float(df[df["category"] == "bg"]["iou"].median()) if (df["category"] == "bg").any() else None,
        }
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


# CLI에서 바로 실행할 때만 argparse를 쓰도록 분기
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Moment-DETR foreground/background errors (worst & missed)")
    parser.add_argument("--pred_path", type=str, required=True,
                        help="Path to predictions (.jsonl), each line has pred_relevant_windows=[[s,e,conf],...]")
    parser.add_argument("--gt_path", type=str, required=True,
                        help="Path to ground truths (.jsonl), each line has relevant_windows=[[gs,ge],...]")
    parser.add_argument("--output_dir", type=str, default="analysis_fg_bg",
                        help="Directory to save outputs")
    parser.add_argument("--conf_thresh", type=float, default=0.3,
                        help="Confidence threshold θ: conf>θ → foreground, else background")
    parser.add_argument("--worst_k", type=int, default=50,
                        help="Top-K to export for each of worst/missed categories")
    parser.add_argument("--fg_iou_bad", type=float, default=0.3,
                        help="IoU upper bound to consider foreground prediction as 'bad'")
    parser.add_argument("--bg_iou_good", type=float, default=0.5,
                        help="IoU lower bound to consider background prediction as 'good/near GT'")

    args = parser.parse_args()

    analyze(pred_path=args.pred_path,
            gt_path=args.gt_path,
            output_dir=args.output_dir,
            conf_thresh=args.conf_thresh,
            worst_k=args.worst_k,
            fg_iou_bad=args.fg_iou_bad,
            bg_iou_good=args.bg_iou_good)
