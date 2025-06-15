import os, json


src_file = "data/raw/FOLIO/folio_v2_validation.jsonl"
tgt_file = "data/cooked/FOLIO.jsonl"
data_src = "FOLIO"

tgt_dir = os.path.dirname(tgt_file)
os.makedirs(tgt_dir, exist_ok=True)

with open(src_file, "r") as src_f, open(tgt_file, "w") as tgt_f:
    cnt = 0
    for line in src_f:
        data = json.loads(line)
        
        example_id = data["example_id"]
        id_ = f"{data_src}-{example_id}"
        premises = data["premises"]
        premises_list = premises.split("\n")
        conclusion = data["conclusion"]

        problem = "Solve the following problem. Your final answer should be \"True\", \"False\", or \"Uncertain\".\n\n"
        problem += "Here a list of premises:\n"
        for i, premise in enumerate(premises_list):
            problem += f"{i+1}. {premise}\n"
        problem += "\n"
        problem += f"According to the premises above, is the following conclusion \"True\", \"False\", or \"Uncertain\"? {conclusion}"
        
        solution = data["label"]
        
        new_data = {
            "id": id_,
            "problem": problem,
            "gt_solution": solution,
            "gt_answer": solution
        }
        
        new_data_str = json.dumps(new_data)
        tgt_f.write(new_data_str + "\n")
        cnt += 1
print(f"Finished writing {cnt} lines to {tgt_file}")