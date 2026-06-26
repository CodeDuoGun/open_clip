"""
python -m open_clip_train.main \
  --dataset-type csv \
  --train-data /Users/tangxueduo/data/tongue/train.csv \
  --val-data /Users/tangxueduo/data/tongue/val.csv \
  --csv-img-key filepath \
  --csv-caption-key title \
  --csv-separator $'\t' \
  --model ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --batch-size 32 \
  --lr 1e-5 \
  --wd 0.1 \
  --epochs 10 \
  --workers 4 \
  --save-frequency 1 \
  --report-to tensorboard \
  --logs ./logs/tongue_vitb32

"""