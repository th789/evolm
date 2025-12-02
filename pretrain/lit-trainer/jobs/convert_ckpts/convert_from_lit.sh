# model_dir=/path/to/pretrained/ckpt

# ----------- exp01: Models that are pretrained on FineWeb -----------
#0.5B models
# model_dir=models/pretrained/llama-0.5B-10BT-weightdecay10.0-seed42
# model_dir=models/pretrained/llama-0.5B-10BT-weightdecay3.0-seed42
# model_dir=models/pretrained/llama-0.5B-10BT-weightdecay1.0-seed42
# model_dir=models/pretrained/llama-0.5B-10BT-weightdecay0.1-seed42
# model_dir=models/pretrained/llama-0.5B-10BT-weightdecay0.01-seed42
# model_dir=models/pretrained/llama-0.5B-10BT-weightdecay0.001-seed42
# model_dir=models/pretrained/llama-0.5B-10BT-weightdecay0.0001-seed42

#1B models
# model_dir=models/pretrained/llama-1B-20BT-weightdecay10.0-seed42
model_dir=models/pretrained/llama-1B-20BT-weightdecay3.0-seed42
# model_dir=models/pretrained/llama-1B-20BT-weightdecay1.0-seed42
# model_dir=models/pretrained/llama-1B-20BT-weightdecay0.1-seed42
# model_dir=models/pretrained/llama-1B-20BT-weightdecay0.01-seed42
# model_dir=models/pretrained/llama-1B-20BT-weightdecay0.001-seed42
# model_dir=models/pretrained/llama-1B-20BT-weightdecay0.0001-seed42

#FINEWEB_FOLDER_PATH_PV path is same for all models above, defined in load_private_vars.sh
# --------------------------------------------------------------------

# ----------- exp02: Models that are pretrained on curated subsets of FineFineWeb -----------
percent_doi=0.001 #options: [0.1, 0.05, 0.01, 0.005, 0.001]

#1B models
# model_dir="models/pretrained/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics${percent_doi}-weightdecay0.0001-seed42"
# model_dir="models/pretrained/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics${percent_doi}-weightdecay0.001-seed42"
# model_dir="models/pretrained/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics${percent_doi}-weightdecay0.01-seed42"
# model_dir="models/pretrained/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics${percent_doi}-weightdecay0.1-seed42"
# model_dir="models/pretrained/ffw/llama-1B-20BT-ffwmysubset20BT-mathematics${percent_doi}-weightdecay1.0-seed42"

#FINEWEB_FOLDER_PATH_PV path is different for each model, defined below
export FINEFINEWEB_FOLDER_PATH_PV="/n/netscratch/doshi-velez_lab/Everyone/ffw_mysubset20BT/mathematics${percent_doi}_litgpt/pretrain"


# -------------------------------------------------------------------------------------------


name=final
lit_ckpt_dir=$model_dir/$name
lit_convert_out_dir=$model_dir/$name-converted
hf_ckpt_dir=$model_dir/$name-hf

litgpt convert_from_litgpt $lit_ckpt_dir $lit_convert_out_dir

python scripts/make_hf_model.py \
    --lit_convert_out_dir $lit_convert_out_dir \
    --save_dir $hf_ckpt_dir \
    --disable_test_vllm

