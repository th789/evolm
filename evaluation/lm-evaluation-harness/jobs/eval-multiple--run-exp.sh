#!/bin/bash

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

zeroshot_tasks="hellaswag,winogrande,piqa,openbookqa,arc_easy,arc_challenge,mathqa"

for wd in "0.0001" \
          "0.001" \
          "0.01" \
          "0.1" \
          "1.0"
do
    ######## exp01
    #pretrained model
    # model_id="/n/home07/than157/desktop/done-large_projects/learn-better/evolm/pretrain/lit-trainer/models/pretrained/llama-1B-20BT-weightdecay${wd}-seed42/final-hf"
    # model_name=$(basename "$(dirname "$model_id")") #second to last folder of model_id
    # subfolder=$(basename "$(dirname "$(dirname "$model_id")")") #third to last folder of model_id
    # OUTPUT_DIR="./eval_output/pretrained/$subfolder/$model_name"

    #finetuned model
    # model_id="/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay${wd}-seed42-metamathqa"
    # model_name=$(basename "$model_id") #last folder of model_id
    # subfolder=$(basename "$(dirname "$model_id")") #second to last folder of model_id
    # OUTPUT_DIR="./eval_output/finetuned/$subfolder/$model_name"
    

    ######## exp02
    #pretrained model
    #0.001
    model_id="/n/home07/than157/desktop/done-large_projects/learn-better/evolm/pretrain/lit-trainer/models/pretrained/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics0.001-weightdecay${wd}-seed42/final-hf"
    # #0.005
    # model_id="/n/home07/than157/desktop/done-large_projects/learn-better/evolm/pretrain/lit-trainer/models/pretrained/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics0.005-weightdecay${wd}-seed42/final-hf"
    # # #0.01
    # model_id="/n/home07/than157/desktop/done-large_projects/learn-better/evolm/pretrain/lit-trainer/models/pretrained/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics0.01-weightdecay${wd}-seed42/final-hf"
    # # #0.05
    # model_id="/n/home07/than157/desktop/done-large_projects/learn-better/evolm/pretrain/lit-trainer/models/pretrained/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics0.05-weightdecay${wd}-seed42/final-hf"
    # # #0.1
    # model_id="/n/home07/than157/desktop/done-large_projects/learn-better/evolm/pretrain/lit-trainer/models/pretrained/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics0.1-weightdecay${wd}-seed42/final-hf"
    # model_name=$(basename "$(dirname "$model_id")") #second to last folder of model_id
    # subfolder=$(basename "$(dirname "$(dirname "$model_id")")") #third to last folder of model_id
    # OUTPUT_DIR="./eval_output/pretrained/$subfolder/$model_name"

    #finetuned model
    #0.001
    # model_id="/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics0.001-weightdecay${wd}-seed42-metamathqa"
    #0.005
    # model_id="/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics0.005-weightdecay${wd}-seed42-metamathqa"
    # #0.01
    # model_id="/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics0.01-weightdecay${wd}-seed42-metamathqa"
    # #0.05
    # model_id="/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics0.05-weightdecay${wd}-seed42-metamathqa"
    # #0.1
    # model_id="/n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics0.1-weightdecay${wd}-seed42-metamathqa"
    model_name=$(basename "$model_id") #last folder of model_id
    subfolder=$(basename "$(dirname "$model_id")") #second to last folder of model_id
    OUTPUT_DIR="./eval_output/finetuned/$subfolder/$model_name"

    echo "---------------------------------------- Evaluating model----------------------------------------"
    echo "Model path: $model_id"
    echo "Model Name: $model_name"
    echo "Weight decay: $wd"
    echo "Output Directory: $OUTPUT_DIR"
    echo "Zero-shot Tasks: $zeroshot_tasks"

    lm_eval --model vllm \
        --model_args pretrained=${model_id},dtype=auto,gpu_memory_utilization=0.7,max_model_len=2048 \
        --tasks $zeroshot_tasks \
        --num_fewshot 0 \
        --output_path "$OUTPUT_DIR" \
        --batch_size auto
    
    echo "Results saved to: $OUTPUT_DIR"

done


end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo ""
echo "Start time:  $(date -d @$start_time)"
echo "End time:    $(date -d @$end_time)"
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))
echo "Elapsed time: ${minutes}min ${seconds}sec"
echo "Complete!"