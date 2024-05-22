#!/bin/sh
#BSUB -J bn_experiment
#BSUB -o job_info/bn_experiment_%J.out
#BSUB -q gpua100
#BSUB -n 16
# #BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1440
#BSUB -R "rusage[mem=16384]"

nvidia-smi

# Load the cuda module
module load cuda/11.6

# Load venv
source ~/context/bin/activate

export CUDA_VISIBLE_DEVICES=0
export TQDM_DISABLE=1 # Clean job info

MODEL_TYPE="r18"
BATCH_SIZE=32
EPOCHS=200
LR_STEP=150

NUM_WORKERS=16

# python train3d.py --model_type $MODEL_TYPE --lr 1e-4 --batch_size $BATCH_SIZE --epochs $EPOCHS --lr_step $LR_STEP --num_workers $NUM_WORKERS --wandb
python train3d.py --model_type $MODEL_TYPE --lr 1e-4 --batch_size $BATCH_SIZE --epochs $EPOCHS --lr_step $LR_STEP --perlin --num_workers $NUM_WORKERS --wandb