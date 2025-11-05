#!/bin/bash

#pass in directory of model to be evaluated as first argument to this bash script
model_ckpt_dir=$1 

#set up env
module load python
mamba activate lm-eval-cot


# OUT_ROOT=/path/to/eval_out/outputs
OUT_ROOT=eval_output

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



# model_list=(
#     zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b
#     zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep1-last100k
#     zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep16-last100k
#     zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep2-last100k
# )

#0.5B models
# PagedAtttention does not support head size = 48 (0.5B models)
# export VLLM_USE_V1=1
# export VLLM_ATTENTION_BACKEND=TORCH_SDPA 
# export VLLM_ENFORCE_EAGER=1

#new
export VLLM_MAX_NUM_BATCHED_TOKENS=4096
model_list=(
    /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.0001-seed42-metamathqa
    # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.001-seed42-metamathqa
    # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.01-seed42-metamathqa
    # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.1-seed42-metamathqa
    # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay1.0-seed42-metamathqa
)


#1B models
# model_list=(
#     /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.0001-seed42-metamathqa
#     /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.001-seed42-metamathqa
#     /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.01-seed42-metamathqa
#     /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.1-seed42-metamathqa
#     /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay1.0-seed42-metamathqa
# )

# for model_ckpt_dir in "${model_list[@]}";
# do

start_time=$(date +%s)
echo "Start time:  $(date -d @$start_time)"

dataset_name=GSM8KPlatinum,MATHLevel1,MATHLevel2,MATHLevel3,MATHLevel4,MATHHard,CRUXEval,BoardgameQA500,TabMWP,StrategyQA500
prompt_config_file=prompts/myllama/default_dataset/prompt_config.json
orm_ckpt_dir=Skywork/Skywork-Reward-Llama-3.1-8B-v0.2

model_id_for_saving=$(basename "$model_ckpt_dir")--greedy
printf "\nRunning collect_responses__greedy, model=$model_ckpt_dir"
collect_responses__greedy ${dataset_name} ${model_ckpt_dir} ${model_id_for_saving} ${prompt_config_file}
# printf "\nRunning evaluate_responses for greedy, model=$model_ckpt_dir"
# evaluate_responses ${dataset_name} ${model_id_for_saving} ${prompt_config_file} ${orm_ckpt_dir}

# model_id_for_saving=$(basename "$model_ckpt_dir")--n16
# printf "\nRunning collect_responses__n16, model=$model_ckpt_dir"
# collect_responses__n16 ${dataset_name} ${model_ckpt_dir} ${model_id_for_saving} ${prompt_config_file}
# printf "\nRunning evaluate_responses for n16, model=$model_ckpt_dir"
# evaluate_responses ${dataset_name} ${model_id_for_saving} ${prompt_config_file} ${orm_ckpt_dir}

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo ""
echo "Start time:  $(date -d @$start_time)"
echo "End time:    $(date -d @$end_time)"
echo "Elapsed time: ${elapsed} seconds"
echo "Complete!"

