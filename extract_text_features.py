#!/usr/bin/env python3
import os
import json
import argparse
import torch
from tqdm import tqdm
import numpy as np
from clip import clip  # ✅ OpenAI CLIP library

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
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_query_only", action="store_true", help="Force using only query text without subtitles")
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
    print(f"[INFO] Extracting token-level text features to {args.output_dir}")

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

        # -----------------------------
        # 🧠 CLIP Transformer로 토큰별 hidden state 추출
        # -----------------------------
        with torch.no_grad():
            tokens = clip.tokenize(text_input).to(device)        # [1, n_tokens]
            x = model.token_embedding(tokens)                    # [1, n_tokens, dim]
            x = x + model.positional_embedding                   # 위치 임베딩 추가
            x = x.permute(1, 0, 2)                               # [n_tokens, 1, dim] (Transformer 입력 순서)
            x = model.transformer(x)                             # [n_tokens, 1, dim]
            x = x.permute(1, 0, 2)                               # [1, n_tokens, dim] (저장용)
            # x = x / x.norm(dim=-1, keepdim=True)               # ⚠️ 필요 시 정규화

        # -----------------------------
        # 💾 저장
        # -----------------------------
        out_path = os.path.join(args.output_dir, f"qid{qid}.npz")
        np.savez(out_path, last_hidden_state=x.squeeze(0).cpu().numpy())  # shape: [1, n_tokens, dim]

    print("[INFO] Done! Token-level features saved to", args.output_dir)


# ------------------------------------------------------------
# 🏁 Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
