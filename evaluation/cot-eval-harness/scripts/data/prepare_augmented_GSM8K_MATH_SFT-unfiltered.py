import os, json


tgt_file = "data/cooked/augmented_GSM8K_MATH_SFT-unfiltered.jsonl"

tgt_dir = os.path.dirname(tgt_file)
os.makedirs(tgt_dir, exist_ok=True)

from datasets import load_dataset

ds = load_dataset("ZhentingNLP/augmented_GSM8K_MATH_SFT-unfiltered", split="default", streaming=True)

our_data = []
for d in ds:
    new_d = {
        "id": d["unique_id"],
        "problem": d["problem"],
        "gt_solution": d["gt_solution"],
        "gt_answer": d["gt_answer"],
    }
    
    our_data.append(new_d)

with open(tgt_file, "w") as tgt_f:
    for d in our_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
        
print(f"Saved {len(our_data)} examples to {tgt_file}")