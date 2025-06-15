export WANDB_API_KEY=<your wandb api key>
export WANDB_ENTITY=<your wandb entity>
export WANDB_PROJECT=<your wandb project>
export CUDA_VISIBLE_DEVICES=0

zeroshot_tasks="hellaswag,winogrande,piqa,openbookqa,arc_easy,arc_challenge,mathqa"

model_id="Qwen/Qwen3-1.7B"
model_name=$(basename "$model_id")

lm_eval --model vllm \
    --model_args pretrained=${model_id},dtype=auto,gpu_memory_utilization=0.6 \
    --tasks $zeroshot_tasks \
    --num_fewshot 0 \
    --wandb_args name=lmeval-0shot-$model_name \
    --batch_size auto

