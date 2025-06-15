export TMPDIR="/path/to/tmp"
export DATA_OPTIMIZER_CACHE_FOLDER="/path/to/tmp"

# Make sure that the second path argument ends with "train"
python litgpt/scripts/prepare_fineweb.py \
    /path/to/fineweb/sample/350BT \
    /path/to/pretrain/train \
    /path/to/hf_ckpts/Llama-2-7b-hf \