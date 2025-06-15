import os, json


tgt_dir = "data/cooked/test"
os.makedirs(tgt_dir, exist_ok=True)

from datasets import load_dataset

ds = load_dataset("tasksource/Boardgame-QA", split="test", streaming=True)

our_data = []
cnt = 0
for d in ds:
    id_ = f"BoardgameQA-{cnt:04d}"
    cnt += 1
    
    question = d['example']
    label = d['label']
    
    if label == "proved":
        solution = "True"
    elif label == "disproved":
        solution = "False"
    elif label == "unknown":
        solution = "Uncertain"
    else:
        raise ValueError(f"Unknown label: {label}")
    
    problem = "Solve the following problem. Your final answer should be \"True\", \"False\", or \"Uncertain\".\n\n"
    problem += f"{question}"
    if len(problem.split()) * 1.3 > 2048:
        continue
    
    new_d = {
        "id": id_,
        "problem": problem,
        "gt_solution": solution,
        "gt_answer": solution
    }
    
    our_data.append(new_d)

tgt_file = os.path.join(tgt_dir, "BoardgameQA.jsonl")
with open(tgt_file, "w") as tgt_f:
    for d in our_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
print(f"Saved {len(our_data)} examples to {tgt_file}")

from random import shuffle
shuffle(our_data)
our_data = our_data[:500]

tgt_file = os.path.join(tgt_dir, "BoardgameQA500.jsonl")
with open(tgt_file, "w") as tgt_f:
    for d in our_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
        
print(f"Saved {len(our_data)} examples to {tgt_file}")