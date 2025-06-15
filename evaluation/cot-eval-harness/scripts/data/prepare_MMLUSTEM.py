import os, json


tgt_dir = "data/cooked/test"
os.makedirs(tgt_dir, exist_ok=True)

from datasets import load_dataset

ds = load_dataset("TIGER-Lab/MMLU-STEM", split="test", streaming=True)

our_data = []
cnt = 0
for d in ds:
    id_ = f"MMLUSTEM-{cnt:04d}"
    cnt += 1
    
    question = d['question']
    choices = d['choices']
    answer = d['answer']
    
    problem = question
    problem += "\n\n"
    assert len(choices) == 4
    mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    for i, c in enumerate(choices):
        problem += f"({mapping[i]}) {c}\n"
    problem += "\n"
    problem += "Please write your final answer in the form of \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
    solution = mapping[answer]
    
    new_d = {
        "id": id_,
        "problem": problem,
        "gt_solution": solution,
        "gt_answer": solution
    }
    
    our_data.append(new_d)

tgt_file = os.path.join(tgt_dir, "MMLUSTEM.jsonl")
with open(tgt_file, "w") as tgt_f:
    for d in our_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
print(f"Saved {len(our_data)} examples to {tgt_file}")

from random import shuffle
shuffle(our_data)
our_data = our_data[:500]

tgt_file = os.path.join(tgt_dir, "MMLUSTEM500.jsonl")
with open(tgt_file, "w") as tgt_f:
    for d in our_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
        
print(f"Saved {len(our_data)} examples to {tgt_file}")