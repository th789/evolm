#!/bin/bash

#pass in arguments to this bash script
#model
model_id=$1 
zeroshot_tasks=$2

start_time=$(date +%s)
echo "Start time:  $(date -d @$start_time)"

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo ""
echo "=== GPU Hardware ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi

#set up env
module load python
module load cuda/12.9.1-fasrc01
mamba activate lm-eval

export CUDA_VISIBLE_DEVICES=0

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export PYTORCH_NVFUSER_DISABLE=1  # if torch.compile uses nvfuser

#datasets
# zeroshot_tasks="hellaswag,winogrande,piqa,openbookqa,arc_easy,arc_challenge,mathqa"

#set up output directory -- based on whether model is pretrained or finetuned
if [[ "$model_id" == *"evolm/pretrain/lit-trainer/models"* ]]; then #pretrained model
    model_name=$(basename "$(dirname "$model_id")") #second to last folder of model_id
    subfolder=$(basename "$(dirname "$(dirname "$model_id")")") #third to last folder of model_id
    OUTPUT_DIR="./eval_output/pretrained/$subfolder/$model_name/$zeroshot_tasks"
elif [[ "$model_id" == *"evolm/finetune/llama-factory/llamafactory_out"* ]]; then #finetuned model
    model_name=$(basename "$model_id") #last folder of model_id
    subfolder=$(basename "$(dirname "$model_id")") #second to last folder of model_id
    OUTPUT_DIR="./eval_output/finetuned/$subfolder/$model_name/$zeroshot_tasks"
fi


lm_eval --model vllm \
    --model_args pretrained=${model_id},dtype=auto,gpu_memory_utilization=0.6,max_model_len=2048,max_num_batched_tokens=4096 \
    --tasks $zeroshot_tasks \
    --num_fewshot 0 \
    --output_path "$OUTPUT_DIR" \
    --batch_size auto


echo "Results saved to: $OUTPUT_DIR"

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo ""
echo "Start time:  $(date -d @$start_time)"
echo "End time:    $(date -d @$end_time)"
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))
echo "Elapsed time: ${minutes}min ${seconds}sec"
echo "Complete!"
