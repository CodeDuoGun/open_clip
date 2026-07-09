"""舌象 CLIP 推理脚本。

两种典型用法：

1) 单张图像检索最匹配的中文描述（图像→文本）：

       python -m medical.infer \\
           --checkpoint ./logs/tongue_vitb32/checkpoints/epoch_10.pt \\
           --image medical/data/tongue_face/<id>.jpg \\
           --model ViT-B-32

2) 对一组候选描述打分，按相似度排序：
       python -m medical.infer \\
           --checkpoint ./logs/tongue_vitb32/checkpoints/epoch_10.pt \\
           --image-dir medical/data/tongue_face \\
           --candidates medical/data/tongue_candidates.txt \\
           --top-k 5 \\
           --output ./logs/tongue_vitb32/predictions.jsonl

``--candidates`` 文件每行一个候选描述；不提供时使用脚本内置的舌象描述池。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


DEFAULT_CANDIDATES_ZH: list[str] = [
    "舌质淡红，薄白苔，舌体正常，无齿痕，无裂纹",
    "舌质淡红，薄白苔，舌体胖大，有齿痕",
    "舌质红，黄腻苔，厚苔",
    "舌质淡白，舌体胖大，有齿痕，苔白腻",
    "舌质紫暗，有瘀斑，苔薄白",
    "舌质红，苔少或无苔，有裂纹",
    "舌质淡红，苔白厚腻，舌体正常",
    "舌质红，苔黄厚腻，舌体胖大",
    "舌下络脉怒张，颜色紫暗",
    "舌质淡红，苔薄白而润，边缘光滑",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot inference with a fine-tuned tongue CLIP.")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        help="open_clip model name (must match the training model).")
    parser.add_argument("--checkpoint", type=str, default='./logs/tongue_vitb32/checkpoints/epoch_10.pt',
                        help="Path to the fine-tuned .pt checkpoint produced by open_clip_train.")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Optional original open_clip tag (e.g. laion2b_s34b_b79k) when --checkpoint "
                             "is a state_dict for finetuning rather than a full model.")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu. Defaults to cuda when available.")
    parser.add_argument("--image", type=str, default="medical/data/tongue_face/bdc54d2b-d75b-2a3c-cbf2-c02272be7726_02134.jpg",
                        help="Single tongue image path.")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory of tongue images to score against the candidates.")
    parser.add_argument("--image-list", type=str, default=None,
                        help="Text file with one image path per line.")
    parser.add_argument("--candidates", type=str, default=None,
                        help="Text file with one candidate description per line.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="When scoring images, output the top-k candidates per image.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSONL output file with per-image predictions.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of images processed (debug).")
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    import torch
    import open_clip

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained #or args.checkpoint,
    )
    tokenizer = open_clip.get_tokenizer(args.model)

    # If --pretrained was provided, override with the trained state_dict on top of those weights.
    if args.pretrained:
        state = torch.load(args.checkpoint, map_location="cpu")
        sd = state.get("state_dict", state) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[warn] missing keys: {len(missing)}; unexpected keys: {len(unexpected)}")

    model = model.to(device).eval()
    return model, preprocess, tokenizer, device


def encode_texts(model, tokenizer, texts: list[str], device: str):
    import torch

    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def encode_image_paths(model, preprocess, image_paths: list[str], device: str):
    import torch
    from PIL import Image

    tensors = []
    for path in image_paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensors.append(preprocess(img))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def iter_images(args: argparse.Namespace):
    if args.image:
        yield args.image
        return
    if args.image_list:
        with open(args.image_list, "r", encoding="utf-8") as fh:
            for line in fh:
                p = line.strip()
                if p:
                    yield p
        return
    if args.image_dir:
        for name in sorted(os.listdir(args.image_dir)):
            full = os.path.join(args.image_dir, name)
            if os.path.isfile(full):
                yield full


def load_candidates(args: argparse.Namespace) -> list[str]:
    if args.candidates:
        with open(args.candidates, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip()]
    return list(DEFAULT_CANDIDATES_ZH)


def topk_indices(scores, k: int) -> list[int]:
    import torch

    k = max(1, min(k, scores.shape[-1]))
    return torch.topk(scores, k=k).indices.tolist()


def main() -> None:
    args = parse_args()
    if not (args.image or args.image_dir or args.image_list):
        raise SystemExit("Provide one of --image / --image-dir / --image-list.")

    model, preprocess, tokenizer, device = load_model(args)
    candidates = load_candidates(args)
    text_feats = encode_texts(model, tokenizer, candidates, device)
    print(f"[infer] device={device}; candidates={len(candidates)}")

    out_fh = open(args.output, "w", encoding="utf-8") if args.output else None

    try:
        if args.image:
            img_feats = encode_image_paths(model, preprocess, [args.image], device)
            scores = (100.0 * img_feats @ text_feats.T).softmax(dim=-1)[0]
            top = topk_indices(scores, args.top_k)
            print(f"\n[image] {args.image}")
            for rank, idx in enumerate(top, start=1):
                print(f"  {rank:>2}. {scores[idx].item():.4f}  {candidates[idx]}")
            if out_fh:
                record = {
                    "image": args.image,
                    "predictions": [
                        {"rank": r, "score": scores[i].item(), "text": candidates[i]}
                        for r, i in enumerate(top, start=1)
                    ],
                }
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            count = 0
            for path in iter_images(args):
                if args.limit is not None and count >= args.limit:
                    break
                img_feats = encode_image_paths(model, preprocess, [path], device)
                scores = (100.0 * img_feats @ text_feats.T).softmax(dim=-1)[0]
                top = topk_indices(scores, args.top_k)
                print(f"\n[image] {path}")
                for rank, idx in enumerate(top, start=1):
                    print(f"  {rank:>2}. {scores[idx].item():.4f}  {candidates[idx]}")
                if out_fh:
                    record = {
                        "image": path,
                        "predictions": [
                            {"rank": r, "score": scores[i].item(), "text": candidates[i]}
                            for r, i in enumerate(top, start=1)
                        ],
                    }
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    finally:
        if out_fh is not None:
            out_fh.close()
            print(f"\n[ok] wrote predictions -> {args.output}")


if __name__ == "__main__":
    main()