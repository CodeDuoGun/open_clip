"""舌象 CLIP 训练脚本。

工作流程：
    1. 调用 ``medical.build_tongue_dataset`` 生成 CSV（若已生成可跳过）。
    2. 调用 ``open_clip_train.main`` 微调一个 CLIP 模型（默认 ViT-B-32）。

用法：

python -m medical.train \
    --model ViT-B-32 \
    --pretrained laion2b_s34b_b79k \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-5 \
    --logs ./logs/tongue_vitb32 \
    --train-csv medical/data/tongue_train.csv \
    --val-csv medical/data/tongue_val.csv \
    --name tongue_vitb32

或仅构建数据集：

    python -m medical.train --build-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MEDICAL_DIR = REPO_ROOT / "medical"
# When this file is executed directly as a script (`python medical/train.py`)
# the parent package is not on sys.path, so ensure both REPO_ROOT and
# MEDICAL_DIR are importable for `medical.*` and module-name lookups.
for path in (REPO_ROOT, MEDICAL_DIR, REPO_ROOT / "src"):
    sp = str(path)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Re-use the dataset builder CLI by spawning it via runpy so its argparse
# remains the single source of truth for build-time options.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune open_clip on the tongue dataset.")
    parser.add_argument("--build-only", action="store_true", help="Only build the CSVs, skip training.")

    parser.add_argument("--annotations", type=Path, default=REPO_ROOT / "tongue_data" / "annotations.jsonl")
    parser.add_argument("--image-root", type=Path, default=REPO_ROOT / "medical" / "data" / "tongue_face")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "medical" / "data")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true", help="Fail when an image is missing.")

    parser.add_argument("--train-csv", type=Path, default=None,
                        help="Override train CSV path. Defaults to <out-dir>/tongue_train.csv.")
    parser.add_argument("--val-csv", type=Path, default=None,
                        help="Override val CSV path. Defaults to <out-dir>/tongue_val.csv.")

    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--precision", type=str, default="amp")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--save-frequency", type=int, default=1)
    parser.add_argument("--logs", type=str, default="./logs/tongue_vitb32")
    parser.add_argument("--name", type=str, default="tongue_vitb32")
    parser.add_argument("--report-to", type=str, default="tensorboard",
                        help="Reporter for open_clip_train.main (tensorboard / wandb / none).")
    parser.add_argument("--resume", type=str, default="latest", help="Checkpoint to resume from.")
    parser.add_argument("--no-amp", action="store_true", help="Shortcut for --precision fp32.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved command line without launching training.")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> tuple[Path, Path]:
    """构造 CSV 数据集，调用 ``build_tongue_dataset`` 的 CLI。"""
    if args.train_csv is not None and args.val_csv is not None:
        if not args.train_csv.is_file():
            raise FileNotFoundError(f"Train CSV not found: {args.train_csv}")
        if not args.val_csv.is_file():
            raise FileNotFoundError(f"Val CSV not found: {args.val_csv}")
        return args.train_csv, args.val_csv

    argv = [
        "--annotations", str(args.annotations),
        "--image-root", str(args.image_root),
        "--out-dir", str(args.out_dir),
        "--val-ratio", str(args.val_ratio),
        "--seed", str(args.seed),
    ]
    if args.strict:
        argv.append("--strict")

    import runpy
    saved_argv = sys.argv
    try:
        sys.argv = ["build_tongue_dataset", *argv]
        runpy.run_module("medical.build_tongue_dataset", run_name="__main__")
    finally:
        sys.argv = saved_argv

    train_csv = args.train_csv or (args.out_dir / "tongue_train.csv")
    val_csv = args.val_csv or (args.out_dir / "tongue_val.csv")
    if not train_csv.is_file():
        raise FileNotFoundError(f"Train CSV not produced: {train_csv}")
    return train_csv, val_csv


def launch_training(args: argparse.Namespace, train_csv: Path, val_csv: Path | None) -> None:
    os.chdir(REPO_ROOT)

    cli: list[str] = [
        "--dataset-type", "csv",
        "--train-data", str(train_csv),
        "--csv-img-key", "filepath",
        "--csv-caption-key", "title",
        "--csv-separator", "\t",
        "--model", args.model,
        "--pretrained", args.pretrained,
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--wd", str(args.wd),
        "--warmup", str(args.warmup),
        "--epochs", str(args.epochs),
        "--workers", str(args.workers),
        "--save-frequency", str(args.save_frequency),
        "--logs", args.logs,
        "--name", args.name,
        "--precision", "fp32" if args.no_amp else args.precision,
        "--report-to", args.report_to,
        "--resume", args.resume,
    ]
    if val_csv is not None and val_csv.is_file():
        cli += ["--val-data", str(val_csv)]

    print("[train] Launching open_clip_train.main with the following args:")
    for token in cli:
        print(f"    {token}")
    if args.dry_run:
        return

    # Import lazily so that --dry-run / --build-only do not require torch/numpy.
    from open_clip_train.main import main as open_clip_main
    open_clip_main(cli)


def main() -> None:
    args = parse_args()
    train_csv, val_csv = build_dataset(args)
    if args.build_only:
        print(f"[ok] build-only finished. Train CSV: {train_csv}, Val CSV: {val_csv}")
        return
    launch_training(args, train_csv, val_csv)


if __name__ == "__main__":
    main()
