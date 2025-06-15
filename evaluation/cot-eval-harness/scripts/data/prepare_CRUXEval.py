import os, json


tgt_file = "data/cooked/test/CRUXEval.jsonl"

tgt_dir = os.path.dirname(tgt_file)
os.makedirs(tgt_dir, exist_ok=True)

from datasets import load_dataset

ds = load_dataset("cruxeval-org/cruxeval", split="test", streaming=True)

our_data = []
cnt = 0
for d in ds:
    id_ = d['id']
    id_ = f"CRUXEval-{id_}"
    cnt += 1
    
    code = d['code']
    input_ = d['input']
    output = d['output']
    
    problem = "You are given the following function:\n\n"
    problem += f"{code}\n\n"
    problem += "You are also given the following input (the order of the input is the same as the order of the input variables in the function):\n\n"
    problem += f"{input_}\n\n"
    problem += "With the given function and input, what would be the output?"
    
    new_d = {
        "id": id_,
        "problem": problem,
        "gt_solution": output,
        "gt_answer": output
    }
    
    our_data.append(new_d)

from random import shuffle
shuffle(our_data)

with open(tgt_file, "w") as tgt_f:
    for d in our_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
        
print(f"Saved {len(our_data)} examples to {tgt_file}")