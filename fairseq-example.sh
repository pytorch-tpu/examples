#!/bin/bash

# Note to user: update the following lines
data_path=/path/to/data
repo_path=/path/to/repo
checkpoints_path=/path/to/saved/checkpoints
export TPU_IP_ADDRESS=10.2.2.2
# Note to user: update the lines above

export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export XLA_USE_BF16=1  # Note: this is optional

python ${repo_path}/deps/fairseq/train.py \
  $data_path \
  --save-dir=${checkpoints_path} \
  --save-interval=1 \
  --arch=transformer_vaswani_wmt_en_de_big \
  --max-target-positions=64 \
  --attention-dropout=0.1 \
  --no-progress-bar \
  --criterion=label_smoothed_cross_entropy \
  --source-lang=en \
  --lr-scheduler=inverse_sqrt \
  --min-lr 1e-09 \
  --skip-invalid-size-inputs-valid-test \
  --target-lang=de \
  --label-smoothing=0.1 \
  --update-freq=1 \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --warmup-init-lr 1e-07 \
  --lr 0.0005 \
  --warmup-updates 4000 \
  --share-all-embeddings \
  --dropout 0.3 \
  --weight-decay 0.0 \
  --valid-subset=valid \
  --train-subset=train \
  --max-epoch=25 \
  --input_shapes 512x32 256x64 640x16 \
  --num_cores=8 \
  --metrics_debug \
  --log_steps=100
