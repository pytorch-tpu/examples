#!/bin/bash

# Before running anyone of these GLUE tasks you should download the
# [GLUE data](https://gluebenchmark.com/tasks) by running
# [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
# and unpack it to some directory `$GLUE_DIR`.

# Note to user: update the following lines.
export TPU_IP_ADDRESS="10.0.0.2"
export MODEL_TYPE="bert"
export MODEL_NAME="bert-base-cased"
export TASK="MRPC"
export GLUE_DIR="/path/to/glue/datasets"
# Note to user: update the above lines.

export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export DATA_DIR="${GLUE_DIR?}/${TASK?}"

python deps/transformers/examples/run_glue_tpu.py \
  --model_type ${MODEL_TYPE?} \
  --model_name_or_path ${MODEL_NAME?} \
  --task_name ${TASK?} \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ${DATA_DIR?} \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/${TASK?} \
  --overwrite_output_dir \
  --logging_steps 50 \
  --save_steps 50 \
  --num_cores=8 \
  --only_log_master
