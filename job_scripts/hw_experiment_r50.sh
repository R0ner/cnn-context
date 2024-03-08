#!/bin/sh
#BSUB -J hw_experiment
#BSUB -o job_info/hw_experiment_%J.out
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -W 240
#BSUB -R "rusage[mem=16384]"

nvidia-smi

# Load the cuda module
module load cuda/11.6

# Load venv
source ~/context/bin/activate

export CUDA_VISIBLE_DEVICES=0
export TQDM_DISABLE=1

python train.py --model_type "r50" --lr 1e-3 --batch_size 16 --wandb