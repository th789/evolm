#!/bin/bash

export WANDB_API_KEY=b10df87569c5fdcef6d7b86acf29819b378fe28d
export WANDB_ENTITY=th789-harvard
export WANDB_PROJECT=overtraining

module load python
mamba activate litgpt-e

litgpt pretrain --config config_hub/custom_configs/pretrain/pythia-14m.yaml