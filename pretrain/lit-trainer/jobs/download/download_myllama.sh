export HF_TOKEN=<your huggingface token>

ckpt_dir=/path/to/downloaded_and_converted_ckpts_from_hf

litgpt download zhenting/myllama-1B-20BT \
    --checkpoint_dir $ckpt_dir \
    --model_name myllama-1B

litgpt download zhenting/myllama-4B-80BT \
    --checkpoint_dir $ckpt_dir \
    --model_name myllama-4B
