#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

DATASET="gibson"

export GLOG_minloglevel=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Copy this file to log location for tracking the flags used.
LOG_LOCATION="output_files/"${TASK}"/"${DATASET}"/splitnet_pretrain"
mkdir -p ${LOG_LOCATION}
cp "$(readlink -f $0)" ${LOG_LOCATION}

#CHECKPOINT_DIRNAME="/coc/testnvme/jtruong33/google_nav/splitnet/output_files/hm3d/splitnet_pretrain/checkpoints/2022_02_05_23_49_41/"
#CHECKPOINT_DIRNAME="/coc/testnvme/jtruong33/google_nav/splitnet/output_files/hm3d/splitnet_pretrain/checkpoints/2022_02_07_18_50_09"
CHECKPOINT_DIRNAME="/coc/testnvme/jtruong33/google_nav/splitnet/output_files/hm3d/splitnet_pretrain/checkpoints/2022_02_09_19_48_49"
python supervised_learning/splitnet_eval.py \
    --encoder-network-type ShallowVisualEncoder \
    --log-prefix ${LOG_LOCATION} \
    --lr 5e-4 \
    --dataset ${DATASET} \
    --data-subset train \
    --num-processes 4 \
    --pytorch-gpu-ids 0 \
    --render-gpu-ids 0 \
    --task pretrain \
    --debug \
    --checkpoint-dirname ${CHECKPOINT_DIRNAME} \

