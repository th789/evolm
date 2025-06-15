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
        --chat_template_file prompts/QwQ-32B-Preview/default_dataset/chat_template.json \
        --tensor_parallel_size 8 \
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

model_ckpt_dir=Qwen/QwQ-32B-Preview
model_id_for_saving=QwQ-32B-Preview--zeroshot
prompt_config_file=prompts/QwQ-32B-Preview/default_dataset/prompt_config.json
dataset_name=TableBench,MMLUProNoMathSTEM

collect_responses ${dataset_name} ${model_ckpt_dir} ${model_id_for_saving} ${prompt_config_file}
evaluate_responses ${dataset_name} ${model_id_for_saving} ${prompt_config_file}
