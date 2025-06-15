import os, json
from copy import deepcopy


src_file = "data/raw/MATH/MATH_test.jsonl"
tgt_file = "data/cooked/test/MATH.jsonl"

tgt_dir = os.path.dirname(tgt_file)
os.makedirs(tgt_dir, exist_ok=True)

data = []
with open(src_file, "r") as src_f:
    cnt = 0
    for line in src_f:
        d = json.loads(line)
        
        unique_id = d["id"]
        id_ = f"MATH-{unique_id}"
        query = d["query"]
        response = d["response"]
        level = d["level"]

        new_d = {
            "id": id_,
            "problem": query,
            "gt_solution": response,
            "gt_answer": None,
            "level": level
        }
        data.append(new_d)
        cnt += 1
print(f"Collected {cnt} examples")

cnt = 0
with open(tgt_file, "w") as tgt_f:
    for d in data:
        d_ = deepcopy(d)
        d_.pop("level")
        d_str = json.dumps(d_)
        tgt_f.write(d_str + "\n")
        cnt += 1
print(f"Saved {cnt} examples to {tgt_file}")

cnt = 0
tgt_file = "data/cooked/test/MATHLevel1.jsonl"
with open(tgt_file, "w") as tgt_f:
    for d in data:
        if d["level"] == "Level 1":
            d_ = deepcopy(d)
            d_.pop("level")
            d_str = json.dumps(d_)
            tgt_f.write(d_str + "\n")
            cnt += 1
print(f"Saved {cnt} examples to {tgt_file}")

cnt = 0
tgt_file = "data/cooked/test/MATHLevel2.jsonl"
with open(tgt_file, "w") as tgt_f:
    for d in data:
        if d["level"] == "Level 2":
            d_ = deepcopy(d)
            d_.pop("level")
            d_str = json.dumps(d_)
            tgt_f.write(d_str + "\n")
            cnt += 1
print(f"Saved {cnt} examples to {tgt_file}")

cnt = 0
tgt_file = "data/cooked/test/MATHLevel3.jsonl"
with open(tgt_file, "w") as tgt_f:
    for d in data:
        if d["level"] == "Level 3":
            d_ = deepcopy(d)
            d_.pop("level")
            d_str = json.dumps(d_)
            tgt_f.write(d_str + "\n")
            cnt += 1
print(f"Saved {cnt} examples to {tgt_file}")

cnt = 0
tgt_file = "data/cooked/test/MATHLevel4.jsonl"
with open(tgt_file, "w") as tgt_f:
    for d in data:
        if d["level"] == "Level 4":
            d_ = deepcopy(d)
            d_.pop("level")
            d_str = json.dumps(d_)
            tgt_f.write(d_str + "\n")
            cnt += 1
print(f"Saved {cnt} examples to {tgt_file}")