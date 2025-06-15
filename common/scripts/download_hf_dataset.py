from huggingface_hub import snapshot_download, hf_hub_download
import os

# ------------- fineweb -------------
# allow_patterns = ["sample/350BT/*"]

# snapshot_download(
#     repo_id="HuggingFaceFW/fineweb-edu",
#     local_dir="/path/to/fineweb",    #! <-- change this to your local dir for saving the dataset
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
#     local_dir="/path/to/finemath",    #! <-- change this to your local dir for saving the dataset
#     repo_type="dataset",
#     force_download=True,
#     allow_patterns=allow_patterns,
#     max_workers=8,
# )
# -----------------------------------
