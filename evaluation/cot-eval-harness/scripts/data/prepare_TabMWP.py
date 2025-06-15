import os, json


src_file = "data/raw/TabMWP/test.jsonl"
tgt_file = "data/cooked/test/TabMWP.jsonl"

tgt_dir = os.path.dirname(tgt_file)
os.makedirs(tgt_dir, exist_ok=True)

all_data = []
with open(src_file, "r") as src_f:
    cnt = 0
    for line in src_f:
        data = json.loads(line)
        
        unique_id = data["idx"]
        id_ = f"TabMWP-{unique_id}"
        question = data["question"]
        table = data["table"]
        answer = data["answer"]
        
        problem = "You are given a table with the following data:\n\n"
        problem += f"{table}\n\n"
        problem += "Based on the table, answer the following question:\n\n"
        problem += f"{question}"
        
        new_data = {
            "id": id_,
            "problem": problem,
            "gt_solution": answer,
            "gt_answer": answer
        }
        all_data.append(new_data)
        cnt += 1

with open(tgt_file, "w") as tgt_f:
    for data in all_data:
        s = json.dumps(data)
        tgt_f.write(f"{s}\n")
        
print(f"Wrote {len(all_data)} examples to {tgt_file}")