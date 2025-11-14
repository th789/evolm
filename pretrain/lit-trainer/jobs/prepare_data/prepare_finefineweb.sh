#!/bin/bash

#SBATCH --job-name=prepare_finefineweb
#SBATCH --partition=seas_compute
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00
#SBATCH --mem=64G
#SBATCH --output=logs/prepare_finefineweb_jobid%j
#SBATCH --error=logs/prepare_finefineweb_jobid%j

export TMPDIR="/n/netscratch/doshi-velez_lab/Everyone/tmp"
export DATA_OPTIMIZER_CACHE_FOLDER="/n/netscratch/doshi-velez_lab/Everyone/tmp_data_optimizer_cache"

module load python
mamba activate litgpt-e

# Make sure that the second path argument ends with "train"
python litgpt/scripts/prepare_finefineweb.py \
    /n/netscratch/doshi-velez_lab/Everyone/ffw_mysubset_20BT \
    /n/netscratch/doshi-velez_lab/Everyone/ffw_mysubset_20BT_litgpt/pretrain/train \
    /n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/meta-llama/Llama-2-7b-hf

# # Make sure that the second path argument ends with "train"
# python litgpt/scripts/prepare_finefineweb.py \
#     /path/to/fineweb/sample/350BT \
#     /path/to/pretrain/train \
#     /path/to/hf_ckpts/Llama-2-7b-hf \