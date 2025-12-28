#!/bin/bash

#SBATCH --job-name=pt_llama4B_80BT
#SBATCH --partition=XXX
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00
#SBATCH --mem=128G
#SBATCH --output=logs/pt_llama4B_80BT%j.out
#SBATCH --error=logs/pt_llama4B_80BT%j.err


export WANDB_API_KEY=your_wandb_api_key
export WANDB_ENTITY=th789-harvard
export WANDB_PROJECT=overtraining

export FINEWEB_FOLDER_PATH_PV="/n/netscratch/doshi-velez_lab/Everyone/fineweb_litgpt/350BT/pretrain"
export FINEFINEWEB_FOLDER_PATH_PV="/n/netscratch/" #can be literally any folder since this experiment does not actually use finefineweb


module load python
mamba activate litgpt-e

litgpt pretrain --config config_hub/custom_configs/pretrain/llama-4B-80BT-weightdecay1.0-seed42.yaml
