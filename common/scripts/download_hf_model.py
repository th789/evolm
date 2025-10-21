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
model_id = "zhenting/myllama-4B-80BT"

# Modify the following paths to your own directory
download_path = os.path.join(model_dir, model_id) #! <-- change this to the path you want to download the model to

print(f"Downloading model {model_id} to {download_path}...")
snapshot_path = snapshot_download(repo_id=model_id, local_dir=download_path, max_workers=16, token=access_token_hf_read)

print(f"\nModel downloaded to: {snapshot_path}")
print("Complete!")