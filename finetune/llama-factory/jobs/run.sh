export WANDB_API_KEY=<your wandb api key>
export WANDB_ENTITY=<your wandb entity>
export WANDB_PROJECT=<your wandb project>
export TMPDIR=/path/to/tmp

FORCE_TORCHRUN=1 llamafactory-cli train /path/to/config/yaml/file.yaml
