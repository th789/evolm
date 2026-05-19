import os
from huggingface_hub import snapshot_download
from private.access import access_token_hf_read

# Set model directory
file_path = os.path.abspath(__file__)
file_dir = os.path.dirname(file_path)
project_dir = os.path.dirname(os.path.dirname(file_dir))
model_dir = f"{project_dir}/models/hf_ckpts"

# Model ID of the model to download
# model_id = "meta-llama/Llama-2-7b-hf" #! <-- change this to the model ID you want to download from HF
# model_id = "zhenting/myllama-1B-20BT"
# model_id = "zhenting/myllama-0.5B-10BT"


#####llama-4B-80BT
# model_id = "th135/llama-4B-80BT-weightdecay0.0-seed42"                               #llama-4B-80BT-wd0.0
# model_id = "zhenting/myllama-4B-80BT"                                               #llama-4B-80BT-wd0.1
# model_id = "hlzhang109/llama-4B-80BT-weightdecay1.0-seed42"                         #llama-4B-80BT-wd1.0

#####olmo-1B, default learning rate (4e-4)
# model_id = "sbordt/OLMo-2-1B-1x-WD0"                                                    #olmo-20x-wd0.0
# model_id = "sbordt/OLMo-2-1B-1x"                                                    #olmo-20x-wd0.1
# model_id = "sbordt/OLMo-2-1B-1x-WD03"                                               #olmo-20x-wd0.3
# model_id = "sbordt/OLMo-2-1B-1x-WD06"                                               #olmo-20x-wd0.6
# model_id = "sbordt/OLMo-2-1B-1x-WD1"                                                #olmo-20x-wd1.0
model_id = "sbordt/OLMo-2-1B-7x-WD0"                                                   #olmo-140x-wd0.0
# model_id = "sbordt/OLMo-2-1B-Decayed-Early"     #now called sbordt/OLMo-2-1B-7x         #olmo-140x-wd0.1
# model_id = "sbordt/OLMo-2-1B-WD1"               #now called sbordt/OLMo-2-1B-7x-WD1     #olmo-140x-wd1.0
# model_id = "sbordt/OLMo-2-1B-7x-WD03"                                                   #olmo-140x-wd0.3

#####olmo-1B, varying learning rate
# model_id = "sbordt/OLMo-2-1B-1x-WD01-LR02"        #olmo-20x-wd0.1-lr2e-4
# model_id = "sbordt/OLMo-2-1B-1x-WD01-LR08"        #olmo-20x-wd0.1-lr8e-4
# model_id = "sbordt/OLMo-2-1B-1x-WD06-LR02"        #olmo-20x-wd0.1-lr2e-4
# model_id = "sbordt/OLMo-2-1B-1x-WD06-LR08"        #olmo-20x-wd0.1-lr8e-4
# model_id = "sbordt/OLMo-2-1B-1x-WD1-LR02"         #olmo-20x-wd1.0-lr2e-4
# model_id = "sbordt/OLMo-2-1B-1x-WD1-LR08"         #olmo-20x-wd1.0-lr8e-4





# Modify the following paths to your own directory
download_path = os.path.join(model_dir, model_id) #! <-- change this to the path you want to download the model to

print(f"Downloading model {model_id} to {download_path}...")
snapshot_path = snapshot_download(repo_id=model_id, local_dir=download_path, max_workers=16, token=access_token_hf_read)

print(f"\nModel downloaded to: {snapshot_path}")
print("Complete!")