# OUT_ROOT=/path/to/eval_out/outputs
OUT_ROOT=eval_output


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
        --batch_size 64 \
        --api hf \
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
        --batch_size 64 \
        --api hf \
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
#     # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/models/hf_ckpts/zhenting/myllama-4B-80BT
#     # zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b
#     # zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep1-last100k
#     # zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep16-last100k
#     # zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep2-last100k
# )

#### ----------- different settings I tried for 0.5B models (BUT DID NOT WORK!-------------
# PagedAtttention does not support head size = 48 (0.5B models)
# export VLLM_USE_V1=1
# export VLLM_ATTENTION_BACKEND=TORCH_SDPA 
# export VLLM_ENFORCE_EAGER=1
# export VLLM_FLASH_ATTN_VERSION=3

# export VLLM_USE_FLEX_ATTENTION=1
# export VLLM_DISABLE_TORCH_COMPILE=1

# export VLLM_USE_FLEX_ATTENTION=1
# export VLLM_DISABLE_TORCH_COMPILE=1
# export VLLM_ENFORCE_EAGER=1

# export VLLM_DISABLE_TORCH_COMPILE=1
# export VLLM_ENFORCE_EAGER=1
# export VLLM_ATTENTION_BACKEND=TORCH_SDPA 

# export VLLM_MAX_NUM_BATCHED_TOKENS=4096
# export VLLM_USE_FLEX_ATTENTION=0
# export VLLM_ATTENTION_BACKEND=TORCH_SDPA
# export VLLM_ENFORCE_EAGER=1

# echo "VLLM_MAX_NUM_BATCHED_TOKENS=$VLLM_MAX_NUM_BATCHED_TOKENS"
# echo "VLLM_USE_FLEX_ATTENTION=$VLLM_USE_FLEX_ATTENTION"
# echo "ATTN_BACKEND=$VLLM_ATTENTION_BACKEND"
# echo "ENFORCE_EAGER=$VLLM_ENFORCE_EAGER"
# echo "DISABLE_TORCH_COMPILE=$VLLM_DISABLE_TORCH_COMPILE"
#### --------------------------------------------------------------------------------------


model_list=(
    /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.0001-seed42-metamathqa
    # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.001-seed42-metamathqa
    # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.01-seed42-metamathqa
    # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay0.1-seed42-metamathqa
    # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-0.5B-10BT-weightdecay1.0-seed42-metamathqa
)


#1B models
# model_list=(
#     # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.0001-seed42-metamathqa
#     # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.001-seed42-metamathqa
#     # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.01-seed42-metamathqa
#     # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay0.1-seed42-metamathqa
#     # /n/home07/than157/desktop/done-large_projects/learn-better/evolm/finetune/llama-factory/llamafactory_out/llama-1B-20BT-weightdecay1.0-seed42-metamathqa
# )

for model_ckpt_dir in "${model_list[@]}";
do
    start_time=$(date +%s)

    # dataset_name=GSM8KPlatinum,MATHLevel1,MATHLevel2,MATHLevel3,MATHLevel4,MATHHard,CRUXEval,BoardgameQA500,TabMWP,StrategyQA500
    # dataset_name=GSM8KPlatinum,MATHLevel1,MATHLevel2,MATHLevel3,MATHLevel4,MATHHard
    dataset_name=MATHLevel1
    prompt_config_file=prompts/myllama/default_dataset/prompt_config.json
    orm_ckpt_dir=Skywork/Skywork-Reward-Llama-3.1-8B-v0.2

    model_id_for_saving=$(basename "$model_ckpt_dir")--greedy
    echo "Start time:  $(date -d @$start_time)"
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

done

