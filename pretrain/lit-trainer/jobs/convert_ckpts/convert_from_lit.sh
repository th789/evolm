# model_dir=/path/to/pretrained/ckpt
# model_dir=models/pretrained/llama-0.5B-10BT-weightdecay1.0-seed42
# model_dir=models/pretrained/llama-0.5B-10BT-weightdecay0.1-seed42
# model_dir=models/pretrained/llama-0.5B-10BT-weightdecay0.01-seed42
# model_dir=models/pretrained/llama-0.5B-10BT-weightdecay0.001-seed42
# model_dir=models/pretrained/llama-0.5B-10BT-weightdecay0.0001-seed42

# model_dir=models/pretrained/llama-1B-20BT-weightdecay1.0-seed42
# model_dir=models/pretrained/llama-1B-20BT-weightdecay0.1-seed42
# model_dir=models/pretrained/llama-1B-20BT-weightdecay0.01-seed42
# model_dir=models/pretrained/llama-1B-20BT-weightdecay0.001-seed42
model_dir=models/pretrained/llama-1B-20BT-weightdecay0.0001-seed42

name=final
lit_ckpt_dir=$model_dir/$name
lit_convert_out_dir=$model_dir/$name-converted
hf_ckpt_dir=$model_dir/$name-hf

litgpt convert_from_litgpt $lit_ckpt_dir $lit_convert_out_dir

python scripts/make_hf_model.py \
    --lit_convert_out_dir $lit_convert_out_dir \
    --save_dir $hf_ckpt_dir \
    --disable_test_vllm

