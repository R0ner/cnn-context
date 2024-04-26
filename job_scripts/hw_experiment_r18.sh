#!/bin/sh
#BSUB -J hw_experiment
#BSUB -o job_info/hw_experiment_%J.out
#BSUB -q gpua100
#BSUB -n 4
# #BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 960
#BSUB -R "rusage[mem=16384]"

nvidia-smi

# Load the cuda module
module load cuda/11.6

# Load venv
source ~/context/bin/activate

export CUDA_VISIBLE_DEVICES=0
export TQDM_DISABLE=1

python train.py --model_type "r18" --lr 1e-4 --batch_size 16 --lr_patience 90 --patience 135 --num_workers 4 --sp_loss --sp_lw "constant" --sp_weight 1 --wandb