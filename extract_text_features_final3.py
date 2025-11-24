#!/usr/bin/env python3
import os
import json
import argparse
import torch
from tqdm import tqdm
import numpy as np
from clip import clip  # OpenAI CLIP library

# ------------------------------------------------------------
# 🎬 [함수] 자막 파일 로드
# ------------------------------------------------------------
def load_subtitles(subs_path):
    subs_dict = {}
    if os.path.exists(subs_path):
        print(f"[INFO] Loading subtitles from {subs_path}")
        with open(subs_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                subs_dict[item["qid"]] = item.get("subtitles", "")
    else:
        print(f"[WARN] No subtitles found at {subs_path}. Continuing with query-only mode.")
    return subs_dict


# ------------------------------------------------------------
# 🚀 [메인 함수]
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to highlight_{split}_release.jsonl")
    parser.add_argument("--subs", default=None, help="Path to subs_{split}.jsonl (optional)")
    parser.add_argument("--output_dir", required=True, help="Directory to save text features")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--use_query_only", action="store_true", help="Use only query text without subtitles")
    args = parser.parse_args()

    # -----------------------------
    # ⚙️ CLIP 모델 로드
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------
    # 🧾 자막 로드
    # -----------------------------
    subs_dict = {}
    if args.subs and not args.use_query_only:
        subs_dict = load_subtitles(args.subs)
    else:
        print("[INFO] Skipping subtitles: use_query_only mode enabled.")

    # -----------------------------
    # 📚 highlight 데이터 로드
    # -----------------------------
    with open(args.jsonl, 'r') as f:
        data = [json.loads(line.strip()) for line in f]

    print(f"[INFO] Loaded {len(data)} entries from {args.jsonl}")
    print(f"[INFO] Extracting sentence-level CLIP text features to {args.output_dir}")

    # -----------------------------
    # 🔁 각 항목별 feature 추출
    # -----------------------------
    for item in tqdm(data):
        qid = item["qid"]
        query = item["query"].strip()
        subs = subs_dict.get(qid, "")

        if args.use_query_only or subs == "":
            text_input = query
        else:
            text_input = query + " " + subs

        with torch.no_grad():
            # CLIP 내부 Transformer로 문장 임베딩 추출
            tokens = clip.tokenize(text_input).to(device)
            x = model.token_embedding(tokens)                      # [1, 77, 512]
            x = x + model.positional_embedding.type(model.dtype)   # pos embed dtype 맞춤
            x = x.permute(1, 0, 2)                                 # [77, 1, 512]
            x = model.transformer(x)                               # [77, 1, 512]
            x = x.permute(1, 0, 2)                                 # [1, 77, 512]
            x = model.ln_final(x)                                  # [1, 77, 512]

            # 🔧 Moment-DETR 호환: (1,77,512) → (77,512) → (512,)
            x = x.squeeze(0)               # batch 차원 제거
            x = x.mean(dim=0)              # 평균 풀링 (문장 임베딩)

        # -----------------------------
        # 💾 저장 (Moment-DETR 호환)
        # -----------------------------
        out_path = os.path.join(args.output_dir, f"qid{qid}.npz")
        np.savez(out_path, last_hidden_state=x.cpu().numpy())  # shape: (512,)

    print("[INFO] Done! Features saved to", args.output_dir)


# ------------------------------------------------------------
# 🏁 Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
