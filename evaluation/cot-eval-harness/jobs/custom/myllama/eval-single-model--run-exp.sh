#!/bin/bash

#pass in arguments to this bash script
# 1. directory of model to be evaluated
# 2. dataset to be evaluated
# 3. output root directory
model_ckpt_dir=$1 
dataset_name=$2 #options: dataset_name=GSM8KPlatinum,MATHLevel1,MATHLevel2,MATHLevel3,MATHLevel4,MATHHard,CRUXEval,BoardgameQA500,TabMWP,StrategyQA500
OUT_ROOT=$3 #options: OUT_ROOT=eval_output, OUT_ROOT=eval_output/ffw

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

# set up env
module load python
module load cuda/12.9.1-fasrc01
mamba activate lm-eval-cot-og


### --------------- NOTE ---------------
# For 0.5B moodels, use these args in collect_responses__greedy() and collect_responses__n16()
# because vllm api cannot handle custom head size = 48
        # --batch_size 64 \
        # --api hf \

# For 1B moodels, use these args in collect_responses__greedy() and collect_responses__n16()
        # --batch_size 500 \
        # --api vllm \
        #can leave out these args because they are the default args in collect_responses.py
### -------------------------------------


# OUT_ROOT=/path/to/eval_out/outputs
# OUT_ROOT=eval_output
# OUT_ROOT=eval_output/ffw

function collect_responses__greedy() {
    dataset_name=$1
    model_ckpt_dir=$2
    model_id_for_saving=$3
    prompt_config_file=$4
    python collect_responses.py \
        --test_dataset_name ${dataset_name} \
        --model_ckpt_dir ${model_ckpt_dir} \
        --model_id_for_saving ${model_id_for_saving} \
        --prompt_config_file ${prompt_config_file} \
        --repetition_penalty 1.1 \
        --temperature 0.0 \
        --num_generations 1 \
        --max_model_len 2048 \
        --apply_chat_template \
        --out_root ${OUT_ROOT} \
        --batch_size 500 \
        --api vllm \
        # --force

}

function collect_responses__n16() {
    dataset_name=$1
    model_ckpt_dir=$2
    model_id_for_saving=$3
    prompt_config_file=$4
    python collect_responses.py \
        --test_dataset_name ${dataset_name} \
        --model_ckpt_dir ${model_ckpt_dir} \
        --model_id_for_saving ${model_id_for_saving} \
        --prompt_config_file ${prompt_config_file} \
        --repetition_penalty 1.1 \
        --temperature 1.0 \
        --num_generations 16 \
        --max_model_len 2048 \
        --apply_chat_template \
        --out_root ${OUT_ROOT} \
        --batch_size 500 \
        --api vllm \
        # --force

}

function evaluate_responses() {
    dataset_name=$1
    model_id_for_saving=$2
    prompt_config_file=$3
    orm_ckpt_dir=$4
    python evaluate_responses.py \
        --test_dataset_name ${dataset_name} \
        --examinee_model_id ${model_id_for_saving} \
        --prompt_config_file ${prompt_config_file} \
        --orm_ckpt_dir ${orm_ckpt_dir} \
        --model_output_root ${OUT_ROOT} \
        # --force

}

export CUDA_VISIBLE_DEVICES=0


#### model is given as argument to this bash script
# example models
# model_list=(
#     zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b
#     zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep1-last100k
#     zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep16-last100k
#     zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep2-last100k
# )


# 0.5B models
# model_list=(
#     /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.0001-seed42-metamathqa
#     # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.001-seed42-metamathqa
#     # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.01-seed42-metamathqa
#     # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.1-seed42-metamathqa
#     # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay1.0-seed42-metamathqa
# )


# 1B models
# model_list=(
#     /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.0001-seed42-metamathqa
#     # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.001-seed42-metamathqa
#     # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.01-seed42-metamathqa
#     # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.1-seed42-metamathqa
#     # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay1.0-seed42-metamathqa
# )





# dataset_name=GSM8KPlatinum,MATHLevel1,MATHLevel2,MATHLevel3,MATHLevel4,MATHHard,CRUXEval,BoardgameQA500,TabMWP,StrategyQA500
prompt_config_file=prompts/myllama/default_dataset/prompt_config.json
orm_ckpt_dir=Skywork/Skywork-Reward-Llama-3.1-8B-v0.2

#greedy
model_id_for_saving=$(basename "$model_ckpt_dir")--greedy
printf "\nRunning collect_responses__greedy, model=$model_ckpt_dir"
collect_responses__greedy ${dataset_name} ${model_ckpt_dir} ${model_id_for_saving} ${prompt_config_file}
printf "\nRunning evaluate_responses for greedy, model=$model_ckpt_dir"
evaluate_responses ${dataset_name} ${model_id_for_saving} ${prompt_config_file} ${orm_ckpt_dir}

#n16
model_id_for_saving=$(basename "$model_ckpt_dir")--n16
printf "\nRunning collect_responses__n16, model=$model_ckpt_dir"
collect_responses__n16 ${dataset_name} ${model_ckpt_dir} ${model_id_for_saving} ${prompt_config_file}
printf "\nRunning evaluate_responses for n16, model=$model_ckpt_dir"
evaluate_responses ${dataset_name} ${model_id_for_saving} ${prompt_config_file} ${orm_ckpt_dir}

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo ""
echo "Start time:  $(date -d @$start_time)"
echo "End time:    $(date -d @$end_time)"
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))
echo "Elapsed time: ${minutes}min ${seconds}sec"
echo "Complete!"

