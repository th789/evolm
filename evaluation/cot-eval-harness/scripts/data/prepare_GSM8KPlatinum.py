import os, json


tgt_file = "data/cooked/GSM8KPlatinum.jsonl"

tgt_dir = os.path.dirname(tgt_file)
os.makedirs(tgt_dir, exist_ok=True)

from datasets import load_dataset

ds = load_dataset("madrylab/gsm8k-platinum", split="test", streaming=True)

our_data = []
cnt = 0
for d in ds:
    id_ = f"GSM8KPlatinum-{cnt:04d}"
    cnt += 1
    
    problem = d['question']
    solution = d['answer']
    
    new_d = {
        "id": id_,
        "problem": problem,
        "gt_solution": solution,
        "gt_answer": None
    }
    
    our_data.append(new_d)

from random import shuffle
shuffle(our_data)

with open(tgt_file, "w") as tgt_f:
    for d in our_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
        
print(f"Saved {len(our_data)} examples to {tgt_file}")