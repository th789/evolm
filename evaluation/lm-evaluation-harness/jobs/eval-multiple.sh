export WANDB_API_KEY=<your wandb api key>
export WANDB_ENTITY=<your wandb entity>
export WANDB_PROJECT=<your wandb project>
export CUDA_VISIBLE_DEVICES=0

zeroshot_tasks="hellaswag,winogrande,piqa,openbookqa,arc_easy,arc_challenge,mathqa"

for model_id in "Qwen/Qwen1.5-4B" \
                "Qwen/Qwen2.5-3B" \
                "Qwen/Qwen3-4B-Base" \
                "google/gemma-3-4b-pt" \
                "meta-llama/Llama-3.2-3B"
do

    model_name=$(basename "$model_id")

    lm_eval --model vllm \
        --model_args pretrained=${model_id},dtype=auto,gpu_memory_utilization=0.7,max_model_len=2048 \
        --tasks $zeroshot_tasks \
        --num_fewshot 0 \
        --wandb_args name=lmeval-0shot-$model_name \
        --batch_size auto

done
