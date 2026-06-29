"""舌象 CLIP 部分层微调脚本。

默认策略：
    - 加载 open_clip 预训练权重。
    - 冻结大部分 image tower，仅放开最后若干 image groups。
    - 冻结大部分 text tower，仅放开最后若干 text layers。

这通常比全参微调更适合小规模医学图文数据：显存更低，过拟合风险更小，
同时仍能适配舌象图像和中文舌象描述。

用法：

python -m medical.train_partial \\
    --model ViT-B-32 \\
    --pretrained laion2b_s34b_b79k \\
    --train-csv medical/data/tongue_train.csv \\
    --val-csv medical/data/tongue_val.csv \\
    --epochs 10 \\
    --batch-size 32

先检查实际传给 open_clip_train 的参数：

python -m medical.train_partial --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MEDICAL_DIR = REPO_ROOT / "medical"
for path in (REPO_ROOT, MEDICAL_DIR, REPO_ROOT / "src"):
    sp = str(path)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Partially fine-tune open_clip on the tongue dataset.")
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
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--save-frequency", type=int, default=1)
    parser.add_argument("--logs", type=str, default="./logs/tongue_vitb32_partial")
    parser.add_argument("--name", type=str, default="tongue_vitb32_partial")
    parser.add_argument("--report-to", type=str, default="tensorboard",
                        help="Reporter for open_clip_train.main (tensorboard / wandb / none).")
    parser.add_argument("--resume", type=str, default="latest", help="Checkpoint to resume from.")
    parser.add_argument("--no-amp", action="store_true", help="Shortcut for --precision fp32.")

    parser.add_argument("--image-unlocked-groups", type=int, default=4,
                        help="Leave the last N image tower layer groups trainable.")
    parser.add_argument("--text-unlocked-layers", type=int, default=4,
                        help="Leave the last N text tower layer groups trainable.")
    parser.add_argument("--full-finetune", action="store_true",
                        help="Disable tower locking and train all parameters, like medical.train.")

    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved command line without launching training.")
    return parser.parse_args(argv)


def build_dataset(args: argparse.Namespace) -> tuple[Path, Path]:
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

    if not args.full_finetune:
        cli += [
            "--lock-image",
            "--lock-image-unlocked-groups", str(args.image_unlocked_groups),
            "--lock-text",
            "--lock-text-unlocked-layers", str(args.text_unlocked_layers),
            "--lock-text-freeze-layer-norm",
        ]

    if val_csv is not None and val_csv.is_file():
        cli += ["--val-data", str(val_csv)]

    print("[train-partial] Launching open_clip_train.main with the following args:")
    for token in cli:
        print(f"    {token}")
    if args.dry_run:
        return

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
