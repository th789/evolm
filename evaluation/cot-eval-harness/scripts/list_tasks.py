import os


data_dir = "data/cooked/test"
jsonl_files = os.listdir(data_dir)
jsonl_files.sort()
print("==> All tasks:")
cnt = 0
for f in jsonl_files:
    if f.endswith(".jsonl"):
        name = f.split(".")[0]
        print(f"{name}")
        cnt += 1