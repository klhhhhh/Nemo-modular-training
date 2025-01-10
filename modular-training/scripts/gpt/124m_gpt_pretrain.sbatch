#!/bin/bash
#SBATCH -A m4410_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=6000
export NVIDIA_PYTORCH_VERSION=24.12

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

srun --export=ALL,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT,NVIDIA_PYTORCH_VERSION=$NVIDIA_PYTORCH_VERSION bash /global/homes/k/klhhhhh/NeMo-modular-training/modular-training/scripts/gpt/124m_gpt_pretrain.sh
