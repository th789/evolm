### Create job-specific cache directories on node's local scratch space
# must run within a slurm job because it uses the SLURM_JOB_ID environment variable

JOBTAG="job${SLURM_JOB_ID}_random${RANDOM}"
CACHE_ROOT="/scratch/$USER/$JOBTAG"
mkdir -p "$CACHE_ROOT"

export HF_HOME="${CACHE_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf_cache/datasets"
export HF_HUB_CACHE="${CACHE_ROOT}/hf_cache/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/hf_cache/transformers"   # optional/legacy, but harmless
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export WANDB_DIR="${CACHE_ROOT}/wandb"


mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TRITON_CACHE_DIR" "$WANDB_DIR"

# (Optional) show what you're using
df -h "$CACHE_ROOT"
echo "CACHE_ROOT=$CACHE_ROOT"

echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "WANDB_DIR=$WANDB_DIR"

