export WANDB_API_KEY='b10df87569c5fdcef6d7b86acf29819b378fe28d'
export WANDB_ENTITY='th789-harvard'
export WANDB_PROJECT='overtraining'
export CUDA_VISIBLE_DEVICES=0

# zeroshot_tasks="hellaswag,winogrande,piqa,openbookqa,arc_easy,arc_challenge,mathqa"
zeroshot_tasks="hellaswag,winogrande"

# model_id="Qwen/Qwen3-1.7B"
model_id="/n/home07/than157/desktop/done-large_projects/learn-better/evolm/pretrain/lit-trainer/models/pretrained/llama-0.5B-10BT-weightdecay0.1-seed42/final-hf"
model_name=$(basename "$model_id")

lm_eval --model vllm \
    --model_args pretrained=${model_id},dtype=auto,gpu_memory_utilization=0.6 \
    --tasks $zeroshot_tasks \
    --num_fewshot 0 \
    --wandb_args name=lmeval-0shot-$model_name \
    --batch_size auto

