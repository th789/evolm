import os, json


tgt_file = "data/cooked/test/ZebraLogicBench.jsonl"

tgt_dir = os.path.dirname(tgt_file)
os.makedirs(tgt_dir, exist_ok=True)

from datasets import load_dataset

ds = load_dataset("allenai/ZebraLogicBench-private", "mc_mode", split="test", streaming=True)

our_data = []

cnt = 0
for d in ds:
    question_id = d['id']
    id_ = f"ZebraLogicBench-{question_id}"
    cnt += 1
    
    puzzle = d['puzzle']
    question = d['question']
    choices = d['choices']
    assert 0 < len(choices) <= 10
    answer = d['answer']
    assert answer in choices and choices.count(answer) == 1
    
    problem = puzzle + "\n\n" + question + "\n\n"
    problem += "Please choose one answer from the following options: "
    for c in choices:
        problem += f"{c}, "
    problem = problem[:-2]  # remove the last comma and space
    problem += "\n\n"
    problem += "Please write your final answer in the form of \\boxed{an option you choose}."
    
    new_d = {
        "id": id_,
        "problem": problem,
        "gt_solution": answer,
        "gt_answer": answer,
    }
    
    our_data.append(new_d)
    
with open(tgt_file, "w") as tgt_f:
    for d in our_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
        
print(f"Saved {len(our_data)} examples to {tgt_file}")

from random import shuffle
shuffle(our_data)
small_data = our_data[:500]

tgt_file = "data/cooked/test/ZebraLogicBench500.jsonl"

with open(tgt_file, "w") as tgt_f:
    for d in small_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")

print(f"Saved {len(small_data)} examples to {tgt_file}")