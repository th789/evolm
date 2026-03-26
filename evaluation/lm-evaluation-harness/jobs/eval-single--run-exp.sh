#!/bin/bash

#pass in arguments to this bash script
#model
model_id=$1 

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

#datasets
zeroshot_tasks="hellaswag,winogrande,piqa,openbookqa,arc_easy,arc_challenge,mathqa"

#set up output directory -- based on whether model is pretrained or finetuned
#pretrained model: llama-0.5B and llama-1B
if [[ "$model_id" == *"evolm/pretrain/lit-trainer/models"* ]]; then 
    model_name=$(basename "$(dirname "$model_id")") #second to last folder of model_id
    OUTPUT_DIR="./eval_output/pretrained/$model_name"
#pretrained model: llama-4B and olmo-1B-20x, olmo-1B-140x
elif [[ "$model_id" == *"evolm/models/hf_ckpts"* ]]; then 
    model_name=$(basename "$model_id") # last folder of model_id
    OUTPUT_DIR="./eval_output/pretrained/$model_name"
#finetuned model: llama-factory
elif [[ "$model_id" == *"evolm/finetune/llama-factory/llamafactory_out"* ]]; then #finetuned model
    model_name=$(basename "$model_id") #last folder of model_id
    OUTPUT_DIR="./eval_output/finetuned/$model_name"
#models finetuned on new non-cot tasks (hellaswag/piqa)
elif [[ "$model_id" == *"models/ft_new_tasks"* ]]; then #finetuned model
    model_name=$(basename "$model_id") #last folder of model_id
    OUTPUT_DIR="./eval_output/finetuned_new_tasks/$model_name"
fi


model_args="pretrained=${model_id},dtype=auto,gpu_memory_utilization=0.5,max_num_batched_tokens=8192"
# Force slow tokenizer to avoid Fast tokenizer config bug for OLMo.
if [[ "$model_id" == *"OLMo"* || "$model_id" == *"olmo"* ]]; then
    model_args="${model_args},tokenizer_mode=slow"
fi


lm_eval --model vllm \
    --model_args "${model_args}" \
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
