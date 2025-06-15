import os, json
import random


src_file = "data/raw/StrategyQA/strategyQA_train.json"

tgt_dir = "data/cooked/test"
os.makedirs(tgt_dir, exist_ok=True)

our_data = []
with open(src_file, "r") as src_f:
    data = json.load(src_f)
    print(f"Loaded {len(data)} examples from {src_file}")

    for d in data:
        qid = d["qid"]
        id_ = f"StrategyQA-{qid}"
        facts = d["facts"]
        question = d["question"]
        answer = str(d["answer"])
        
        problem = "You are given the following facts:\n"
        for i, fact in enumerate(facts):
            problem += f"{i+1}. {fact}\n"
        problem += "\n"
        problem += "Based on the facts above, answer the following question. Your final answer should be either \"True\" or \"False\".\n"
        problem += f"{question}"
        
        new_d = {
            "id": id_,
            "problem": problem,
            "gt_solution": answer,
            "gt_answer": answer
        }

        our_data.append(new_d)

tgt_file = os.path.join(tgt_dir, "StrategyQA.jsonl")
with open(tgt_file, "w") as tgt_f:
    for d in our_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
print(f"Saved {len(our_data)} examples to {tgt_file}")

random.shuffle(our_data)
our_data = our_data[:500]

tgt_file = os.path.join(tgt_dir, "StrategyQA500.jsonl")
with open(tgt_file, "w") as tgt_f:
    for d in our_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")

print(f"Saved {len(our_data)} examples to {tgt_file}")