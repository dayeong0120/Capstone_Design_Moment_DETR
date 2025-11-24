#!/usr/bin/env python3
import os
import json
import argparse
import torch
from tqdm import tqdm
import numpy as np
from clip import clip  # OpenAI CLIP library

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to highlight_{split}_release.jsonl")
    parser.add_argument("--subs", default=None, help="Path to subs_{split}.jsonl (optional)")
    parser.add_argument("--output_dir", required=True, help="Directory to save text features")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_query_only", action="store_true", help="Force using only query text without subtitles")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device=device)

    os.makedirs(args.output_dir, exist_ok=True)

    # load data
    subs_dict = {}
    if args.subs and not args.use_query_only:
        subs_dict = load_subtitles(args.subs)
    else:
        print("[INFO] Skipping subtitles: use_query_only mode enabled.")

    with open(args.jsonl, 'r') as f:
        data = [json.loads(line.strip()) for line in f]

    print(f"[INFO] Loaded {len(data)} entries from {args.jsonl}")
    print(f"[INFO] Extracting text features to {args.output_dir}")

    for item in tqdm(data):
        qid = item["qid"]
        query = item["query"].strip()
        subs = subs_dict.get(qid, "")
        if args.use_query_only or subs == "":
            text_input = query
        else:
            text_input = query + " " + subs

        with torch.no_grad():
            tokens = clip.tokenize(text_input).to(device)
            text_feat = model.encode_text(tokens)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)

        out_path = os.path.join(args.output_dir, f"qid{qid}.npz")
        np.savez(out_path, last_hidden_state=text_feat.cpu().numpy().reshape(1, -1))

    print("[INFO] Done! Text features saved to", args.output_dir)


if __name__ == "__main__":
    main()
