import os, json


src_file = "data/raw/MinervaMath/test.jsonl"
tgt_file = "data/cooked/test/MinervaMath.jsonl"

tgt_dir = os.path.dirname(tgt_file)
os.makedirs(tgt_dir, exist_ok=True)

with open(src_file, "r") as src_f, open(tgt_file, "w") as tgt_f:
    cnt = 0
    for line in src_f:
        data = json.loads(line)
        idx = data["idx"]
        
        id_ = f"MinervaMath-{idx}"
        problem = data["problem"]
        solution = data["solution"]
        new_data = {
            "id": id_,
            "problem": problem,
            "gt_solution": solution,
            "gt_answer": None
        }
        new_data_str = json.dumps(new_data)
        tgt_f.write(new_data_str + "\n")
        cnt += 1
print(f"Finished writing {cnt} lines to {tgt_file}")