import os, json


for x in ["main", "diamond"]:
    tgt_file = f"data/cooked/test/GPQA{x.capitalize()}.jsonl"

    tgt_dir = os.path.dirname(tgt_file)
    os.makedirs(tgt_dir, exist_ok=True)

    from datasets import load_dataset

    ds = load_dataset(f"hendrydong/gpqa_{x}_mc", split="test", streaming=True)

    our_data = []
    cnt = 0
    for d in ds:
        id_ = f"GPQA{x.capitalize()}-{cnt:04d}"
        cnt += 1
        
        problem = d['problem']
        solution = d['solution']
        
        assert solution.startswith("\\boxed{")
        assert solution.endswith("}")
        solution = solution.lstrip("\\boxed{").rstrip("}")
        
        new_d = {
            "id": id_,
            "problem": problem,
            "gt_solution": solution,
            "gt_answer": solution
        }
        
        our_data.append(new_d)

    with open(tgt_file, "w") as tgt_f:
        for d in our_data:
            d_str = json.dumps(d)
            tgt_f.write(d_str + "\n")
            
    print(f"Saved {len(our_data)} examples to {tgt_file}")
