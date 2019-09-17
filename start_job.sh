#!/bin/bash
#
#SBATCH --job-name=RL_seq2seq
#SBATCH --partition=pascalGPU
#SBATCH --exclude=i13hpc57
#SBATCH --export=CUDA_HOME=/usr/local/cuda,LD_LIBRARY_PATH=/usr/local/cuda/lib64
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=6000
#SBATCH --cpus-per-task=2
#SBATCH --time=4-23:59:58
#SBATCH --output=train_log.out
srun -u PYTHON_PATH train.py
