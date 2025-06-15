import os, json


for x in [2023, 2024]:
    src_file = f"data/raw/AIME/AIME{x}_test.jsonl"
    tgt_file = f"data/cooked/test/AIME{x}.jsonl"

    tgt_dir = os.path.dirname(tgt_file)
    os.makedirs(tgt_dir, exist_ok=True)

    with open(src_file, "r") as src_f, open(tgt_file, "w") as tgt_f:
        cnt = 0
        for line in src_f:
            data = json.loads(line)
            
            unique_id = data["id"]
            id_ = f"AIME{x}-{unique_id}"
            query = data["query"]
            response = str(data["response"])
            answer = str(data["answer"])
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

