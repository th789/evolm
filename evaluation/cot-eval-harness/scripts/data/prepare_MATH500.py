import os, json


src_file = "data/raw/MATH/MATH500_test.jsonl"
tgt_file = "data/cooked/MATH500_test.jsonl"

tgt_dir = os.path.dirname(tgt_file)
os.makedirs(tgt_dir, exist_ok=True)

with open(src_file, "r") as src_f, open(tgt_file, "w") as tgt_f:
    cnt = 0
    for line in src_f:
        data = json.loads(line)
        
        unique_id = data["unique_id"]
        id_ = f"MATH500-{unique_id.replace('test/', '').replace('/', '-').replace('.json', '')}"
        query = data["query"]
        response = data["response"]
        answer = data["answer"]
        new_data = {
            "id": id_,
            "problem": query,
            "gt_solution": response,
            "gt_answer": answer
        }
        new_data_str = json.dumps(new_data)
        tgt_f.write(new_data_str + "\n")
        cnt += 1
print(f"Finished writing {cnt} lines to {tgt_file}")