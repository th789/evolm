model_dir=/path/to/pretrained/ckpt

name=final
lit_ckpt_dir=$model_dir/$name
lit_convert_out_dir=$model_dir/$name-converted
hf_ckpt_dir=$model_dir/$name-hf

litgpt convert_from_litgpt $lit_ckpt_dir $lit_convert_out_dir

python scripts/make_hf_model.py \
    --lit_convert_out_dir $lit_convert_out_dir \
    --save_dir $hf_ckpt_dir \
    --disable_test_vllm

