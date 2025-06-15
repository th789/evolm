OUT_ROOT=/path/to/eval_out/outputs

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

model_list=(
    zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b
    zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep1-last100k
    zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep16-last100k
    zhenting/myllama-4B-160BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep2-last100k
)

for model_ckpt_dir in "${model_list[@]}";
do

    dataset_name=GSM8KPlatinum,MATHLevel1,MATHLevel2,MATHLevel3,MATHLevel4,MATHHard,CRUXEval,BoardgameQA500,TabMWP,StrategyQA500
    prompt_config_file=prompts/myllama/default_dataset/prompt_config.json
    orm_ckpt_dir=Skywork/Skywork-Reward-Llama-3.1-8B-v0.2

    model_id_for_saving=$(basename "$model_ckpt_dir")--greedy
    collect_responses__greedy ${dataset_name} ${model_ckpt_dir} ${model_id_for_saving} ${prompt_config_file}
    evaluate_responses ${dataset_name} ${model_id_for_saving} ${prompt_config_file} ${orm_ckpt_dir}

    model_id_for_saving=$(basename "$model_ckpt_dir")--n16
    collect_responses__n16 ${dataset_name} ${model_ckpt_dir} ${model_id_for_saving} ${prompt_config_file}
    evaluate_responses ${dataset_name} ${model_id_for_saving} ${prompt_config_file} ${orm_ckpt_dir}

done
