export WANDB_API_KEY=<your wandb api key>
export WANDB_ENTITY=<your wandb entity>
export WANDB_PROJECT=<your wandb project>

CUDA_VISIBLE_DEVICES=0,1,2,3 litgpt pretrain --config /path/to/config/yaml/file.yaml
