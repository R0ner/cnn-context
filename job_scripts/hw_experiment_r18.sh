#!/bin/sh
#BSUB -J hw_experiment
#BSUB -o job_info/hw_experiment_%J.out
#BSUB -q gpuv100
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


# python train.py --model_type "r18" --lr 1e-4 --batch_size 16 --lr_patience 50 --patience 100 --warmup 300 --num_workers 4 --smooth_mode "ma" --n_smooth 10 --wandb
# python train.py --model_type "r18" --lr 1e-4 --batch_size 16 --lr_patience 50 --patience 100 --warmup 300 --num_workers 4 --sp_loss --sp_lw "constant" --sp_weight 0.1 --sp_normalize --sp_binary --sp_binary_th 0.5 --smooth_mode "ma" --n_smooth 10 --wandb
# python train.py --model_type "r18" --lr 1e-4 --batch_size 16 --lr_patience 50 --patience 100 --warmup 300 --num_workers 4 --sp_loss --sp_lw "geometric" --sp_weight 0.1 --sp_normalize --sp_binary --sp_binary_th 0.5 --smooth_mode "ma" --n_smooth 10 --wandb
# python train.py --model_type "r18" --lr 1e-4 --batch_size 16 --lr_patience 50 --patience 100 --warmup 300 --num_workers 4 --sp_loss --sp_lw "geometric" --sp_weight 0.1 --sp_normalize --smooth_mode "ma" --n_smooth 10 --wandb
# python train.py --model_type "r18" --lr 1e-4 --batch_size 16 --lr_patience 50 --patience 100 --warmup 300 --num_workers 4 --sp_loss --sp_lw "geometric" --sp_weight 10 --sp_normalize --smooth_mode "ma" --n_smooth 10 --wandb

# python train.py --model_type "r18" --lr 1e-4 --batch_size 16 --lr_patience 50 --patience 100 --warmup 300 --num_workers 4 --sp_loss --sp_lw "geometric" --sp_weight 1 --sp_normalize --sp_mode "l1" --smooth_mode "ma" --n_smooth 10 --wandb
# python train.py --model_type "r18" --lr 1e-4 --batch_size 16 --lr_patience 50 --patience 100 --warmup 300 --num_workers 4 --sp_loss --sp_lw "constant" --sp_weight 1 --sp_normalize --sp_mode "l1" --smooth_mode "ma" --n_smooth 10 --wandb
# python train.py --model_type "r18" --lr 1e-4 --batch_size 16 --lr_patience 50 --patience 100 --warmup 300 --num_workers 4 --sp_loss --sp_lw "geometric" --sp_weight 1 --sp_normalize --sp_binary --sp_mode "l1" --smooth_mode "ma" --n_smooth 10 --wandb
# python train.py --model_type "r18" --lr 1e-4 --batch_size 16 --lr_patience 50 --patience 100 --warmup 300 --num_workers 4 --sp_loss --sp_lw "constant" --sp_weight 1 --sp_normalize --sp_binary --sp_mode "l1" --smooth_mode "ma" --n_smooth 10 --wandb

# python train.py --model_type "r18" --lr 1e-4 --batch_size 16 --lr_patience 50 --patience 100 --warmup 300 --num_workers 4 --sp_loss --sp_lw "geometric" --sp_weight 0.1 --sp_normalize --sp_mode "l1" --smooth_mode "ma" --n_smooth 10 --wandb

MODEL_TYPE="r18"
BATCH_SIZE=16

LR_PAT=100 # Learning rate patience
PAT=150 # Early stopping patience
WARMUP=250 # N warmup epochs before reducing lr and doing early stopping
SMOOTH_MODE="ma" # Smoothing for learnig rate scheduling and early stopping
N_SMOOTH=10 # no. steps to include in smoothing (see "smooth_mode") 

NUM_WORKERS=4


# Experiments
# Cross entropy loss
# python train.py --model_type $MODEL_TYPE --lr 1e-4 --batch_size $BATCH_SIZE --lr_patience $LR_PAT --patience $PAT --warmup $WARMUP --smooth_mode $SMOOTH_MODE --n_smooth $N_SMOOTH --num_workers $NUM_WORKERS --wandb
# python train.py --model_type $MODEL_TYPE --lr 1e-4 --wd 1e-3 --batch_size $BATCH_SIZE --lr_patience $LR_PAT --patience $PAT --warmup $WARMUP --smooth_mode $SMOOTH_MODE --n_smooth $N_SMOOTH --num_workers $NUM_WORKERS --wandb

# Cross entropy loss + "superpixel loss"
# L2 norm
# Constant layer weights
# python train.py --model_type $MODEL_TYPE --lr 1e-4 --batch_size $BATCH_SIZE --sp_loss --sp_lw "constant" --sp_weight 1 --sp_normalize --sp_mode "l2" --lr_patience $LR_PAT --patience $PAT --warmup $WARMUP --smooth_mode $SMOOTH_MODE --n_smooth $N_SMOOTH --num_workers $NUM_WORKERS --wandb
# python train.py --model_type $MODEL_TYPE --lr 1e-4 --batch_size $BATCH_SIZE --sp_loss --sp_lw "constant" --sp_weight 0.1 --sp_normalize --sp_mode "l2" --lr_patience $LR_PAT --patience $PAT --warmup $WARMUP --smooth_mode $SMOOTH_MODE --n_smooth $N_SMOOTH --num_workers $NUM_WORKERS --wandb

# Geometric layer weights 
# python train.py --model_type $MODEL_TYPE --lr 1e-4 --batch_size $BATCH_SIZE --sp_loss --sp_lw "geometric" --sp_weight 1 --sp_normalize --sp_mode "l2" --lr_patience $LR_PAT --patience $PAT --warmup $WARMUP --smooth_mode $SMOOTH_MODE --n_smooth $N_SMOOTH --num_workers $NUM_WORKERS --wandb
# python train.py --model_type $MODEL_TYPE --lr 1e-4 --batch_size $BATCH_SIZE --sp_loss --sp_lw "geometric" --sp_weight 0.1 --sp_normalize --sp_mode "l2" --lr_patience $LR_PAT --patience $PAT --warmup $WARMUP --smooth_mode $SMOOTH_MODE --n_smooth $N_SMOOTH --num_workers $NUM_WORKERS --wandb

# L1 norm
# Constant layer weights
# python train.py --model_type $MODEL_TYPE --lr 1e-4 --batch_size $BATCH_SIZE --sp_loss --sp_lw "constant" --sp_weight 1 --sp_normalize --sp_mode "l1" --lr_patience $LR_PAT --patience $PAT --warmup $WARMUP --smooth_mode $SMOOTH_MODE --n_smooth $N_SMOOTH --num_workers $NUM_WORKERS --wandb
# python train.py --model_type $MODEL_TYPE --lr 1e-4 --batch_size $BATCH_SIZE --sp_loss --sp_lw "constant" --sp_weight 0.1 --sp_normalize --sp_mode "l1" --lr_patience $LR_PAT --patience $PAT --warmup $WARMUP --smooth_mode $SMOOTH_MODE --n_smooth $N_SMOOTH --num_workers $NUM_WORKERS --wandb

# Geometric layer weights
# python train.py --model_type $MODEL_TYPE --lr 1e-4 --batch_size $BATCH_SIZE --sp_loss --sp_lw "geometric" --sp_weight 1 --sp_normalize --sp_mode "l1" --lr_patience $LR_PAT --patience $PAT --warmup $WARMUP --smooth_mode $SMOOTH_MODE --n_smooth $N_SMOOTH --num_workers $NUM_WORKERS --wandb
# python train.py --model_type $MODEL_TYPE --lr 1e-4 --batch_size $BATCH_SIZE --sp_loss --sp_lw "geometric" --sp_weight 0.1 --sp_normalize --sp_mode "l1" --lr_patience $LR_PAT --patience $PAT --warmup $WARMUP --smooth_mode $SMOOTH_MODE --n_smooth $N_SMOOTH --num_workers $NUM_WORKERS --wandb

# Weight decay (?)
python train.py --model_type $MODEL_TYPE --lr 1e-4 --wd 1e-3 --batch_size $BATCH_SIZE --sp_loss --sp_lw "constant" --sp_weight 1 --sp_normalize --sp_mode "l1" --lr_patience $LR_PAT --patience $PAT --warmup $WARMUP --smooth_mode $SMOOTH_MODE --n_smooth $N_SMOOTH --num_workers $NUM_WORKERS --wandb