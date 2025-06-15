import os
from huggingface_hub import snapshot_download

model_dir = "/path/to/hf_ckpts"

# Model ID of the model to download
model_id = "meta-llama/Llama-2-7b-hf" #! <-- change this to the model ID you want to download from HF

# Modify the following paths to your own directory
download_path = os.path.join(model_dir, model_id) #! <-- change this to the path you want to download the model to

snapshot_path = snapshot_download(repo_id=model_id, local_dir=download_path, max_workers=16, token="")

print(f"Model downloaded to: {snapshot_path}")