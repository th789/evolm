function collect_responses() {
    dataset_name=$1
    model_ckpt_dir=$2
    model_id_for_saving=$3
    prompt_config_file=$4
    chunk_idx=$5
    total_chunks=$6
    python collect_responses.py \
        --test_dataset_name ${dataset_name} \
        --chunk_idx ${chunk_idx} \
        --total_chunks ${total_chunks} \
        --model_ckpt_dir ${model_ckpt_dir} \
        --model_id_for_saving ${model_id_for_saving} \
        --prompt_config_file ${prompt_config_file} \
        --apply_chat_template \
        --temperature 1.0 \
        --num_generations 10 \
        --force

}

function evaluate_responses() {
    dataset_name=$1
    model_id_for_saving=$2
    prompt_config_file=$3
    chunk_idx=$4
    total_chunks=$5
    python evaluate_responses.py \
        --test_dataset_name ${dataset_name} \
        --chunk_idx ${chunk_idx} \
        --total_chunks ${total_chunks} \
        --examinee_model_id ${model_id_for_saving} \
        --prompt_config_file ${prompt_config_file} \
        --force

}

model_ckpt_dir=Qwen/Qwen2.5-Math-1.5B-Instruct
model_id_for_saving=Qwen2.5-Math-1.5B-Instruct--zeroshot
prompt_config_file=prompts/Qwen2.5-Math-1.5B-Instruct/default_dataset/prompt_config.json
dataset_name=MATH500

collect_responses ${dataset_name} ${model_ckpt_dir} ${model_id_for_saving} ${prompt_config_file} 0 1
evaluate_responses ${dataset_name} ${model_id_for_saving} ${prompt_config_file} 0 1
