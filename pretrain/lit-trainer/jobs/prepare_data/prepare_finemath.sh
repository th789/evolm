export TMPDIR="/path/to/tmp"   #! <-- change this to your tmp dir
export DATA_OPTIMIZER_CACHE_FOLDER="/path/to/tmp" #! <-- change this to your tmp dir

# Make sure that the second path argument ends with "train"
python litgpt/scripts/prepare_finemath.py \
    /path/to/finemath \
    /path/to/data/continued_pretrain/finemath/train \
    /path/to/hf_ckpts/Llama-2-7b-hf
