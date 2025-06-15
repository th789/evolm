## model config
original_model_path=zhenting/myllama-4B-320BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b
trained_model_path=/path/to/verl_out/checkpoints/myllama-4B-320BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b-rlep8-last100k
out_root=/path/to/verl_out/merged_checkpoints
step=384

# get model name
model_name=$(basename $trained_model_path)
target_dir=${out_root}/${model_name}

python scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path $original_model_path \
    --local_dir ${trained_model_path}/global_step_$step/actor \
    --target_dir ${target_dir}

cp ${trained_model_path}/global_step_$step/actor/huggingface/* ${target_dir}/
