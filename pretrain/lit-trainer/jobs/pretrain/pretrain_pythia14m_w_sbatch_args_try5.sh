#!/bin/bash

#SBATCH --job-name=pretrain_pythia14m
#SBATCH --partition=seas_gpu,serial_requeue,gpu_requeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --time=0-02:00
#SBATCH --mem=64G
#SBATCH --output=logs/pretrain_pythia14m_jobid%j.out
#SBATCH --error=logs/pretrain_pythia14m_jobid%j.err


export WANDB_API_KEY=b10df87569c5fdcef6d7b86acf29819b378fe28d
export WANDB_ENTITY=th789-harvard
export WANDB_PROJECT=overtraining

module load python
mamba activate litgpt-e

srun litgpt pretrain --config /n/home07/than157/desktop/done-large_projects/learn-better/evolm/pretrain/lit-trainer/config_hub/custom_configs/pretrain/pythia-14m.yaml