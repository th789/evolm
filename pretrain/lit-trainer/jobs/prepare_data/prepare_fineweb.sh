#!/bin/bash

#SBATCH --job-name=prepare_fineweb
#SBATCH --partition=seas_compute
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00
#SBATCH --mem=64G
#SBATCH --output=logs/prepare_fineweb_jobid%j
#SBATCH --error=logs/prepare_fineweb_jobid%j

export TMPDIR="/n/netscratch/doshi-velez_lab/Everyone/tmp"
export DATA_OPTIMIZER_CACHE_FOLDER="/n/netscratch/doshi-velez_lab/Everyone/tmp_data_optimizer_cache"

module load python
mamba activate litgpt-e

# Make sure that the second path argument ends with "train"
python litgpt/scripts/prepare_fineweb.py \
    /n/netscratch/doshi-velez_lab/Everyone/fineweb/sample/350BT \
    /n/netscratch/doshi-velez_lab/Everyone/fineweb_litgpt/350BT/pretrain/train \
    /n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/meta-llama/Llama-2-7b-hf