function collect_responses() {
    dataset_name=$1
    model_ckpt_dir=$2
    model_id_for_saving=$3
    prompt_config_file=$4
    python collect_responses.py \
        --test_dataset_name ${dataset_name} \
        --model_ckpt_dir ${model_ckpt_dir} \
        --model_id_for_saving ${model_id_for_saving} \
        --prompt_config_file ${prompt_config_file} \
        --apply_chat_template \
        --force

}

function evaluate_responses() {
    dataset_name=$1
    model_id_for_saving=$2
    prompt_config_file=$3
    python evaluate_responses.py \
        --test_dataset_name ${dataset_name} \
        --examinee_model_id ${model_id_for_saving} \
        --prompt_config_file ${prompt_config_file} \
        --force

}

model_ckpt_dir=Qwen/Qwen2.5-7B-Instruct
model_id_for_saving=Qwen2.5-7B-Instruct--zeroshot
prompt_config_file=prompts/Qwen2.5-7B-Instruct/default_dataset/prompt_config.json
dataset_name=MATH500

collect_responses ${dataset_name} ${model_ckpt_dir} ${model_id_for_saving} ${prompt_config_file}
evaluate_responses ${dataset_name} ${model_id_for_saving} ${prompt_config_file}
