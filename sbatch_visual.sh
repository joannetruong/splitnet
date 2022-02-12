#!/bin/bash
#SBATCH --job-name=splitnet_visual_gray
#SBATCH --output=logs/splitnet_gray.out
#SBATCH --error=logs/splitnet_gray.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 7
#SBATCH --partition=long
#SBATCH --chdir /coc/testnvme/jtruong33/google_nav/splitnet

DATASET="hm3d"

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
export PYTHONPATH='/coc/testnvme/jtruong33/google_nav/splitnet'

srun /nethome/jtruong33/miniconda3/envs/habitat-outdoor/bin/python supervised_learning/splitnet_pretrain.py \
    --encoder-network-type ShallowVisualEncoder \
    --log-prefix ${LOG_LOCATION} \
    --lr 5e-4 \
    --dataset ${DATASET} \
    --data-subset train \
    --num-processes 4 \
    --pytorch-gpu-ids 0 \
    --render-gpu-ids 0 \
    --task pretrain \
    #--debug \
