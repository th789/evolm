from huggingface_hub import snapshot_download, hf_hub_download
import os
from private.access import storage_dir

print("Starting data download...")
print("Data will be saved to:", storage_dir)

# ------------- fineweb -------------
# allow_patterns = ["sample/350BT/*"]
# # allow_patterns = ["sample/100BT/*"]

# snapshot_download(
#     repo_id="HuggingFaceFW/fineweb-edu",
#     local_dir=f"{storage_dir}/fineweb",    #! <-- change this to your local dir for saving the dataset
#     repo_type="dataset",
#     force_download=True,
#     allow_patterns=allow_patterns,
#     max_workers=8, # 64
# )
# -----------------------------------

# ------------- finemath -------------
# allow_patterns = ["finemath-3plus/*", "infiwebmath-3plus/*"]

# snapshot_download(
#     repo_id="HuggingFaceTB/finemath",
#     local_dir=f"{storage_dir}/finemath",    #! <-- change this to your local dir for saving the dataset
#     repo_type="dataset",
#     force_download=True,
#     allow_patterns=allow_patterns,
#     max_workers=8,
# )
# -----------------------------------

print("\n Data download complete!")