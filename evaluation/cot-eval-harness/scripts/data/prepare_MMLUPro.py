import os, json


tgt_file = "data/cooked/test/MMLUPro.jsonl"

tgt_dir = os.path.dirname(tgt_file)
os.makedirs(tgt_dir, exist_ok=True)

from datasets import load_dataset

ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", streaming=True)

our_data = []
subfield_data = {
    "math": [],
    "physics": [],
    "chemistry": [],
    "law": [],
    "engineering": [],
    "economics": [],
    "health": [],
    "psychology": [],
    "business": [],
    "biology": [],
    "philosophy": [],
    "computer science": [],
    "history": [],
}
subfield2name = {
    "math": "Math",
    "physics": "Physics",
    "chemistry": "Chemistry",
    "law": "Law",
    "engineering": "Engineering",
    "economics": "Economics",
    "health": "Health",
    "psychology": "Psychology",
    "business": "Business",
    "biology": "Biology",
    "philosophy": "Philosophy",
    "computer science": "ComputerScience",
    "history": "History",
}

cnt = 0
for d in ds:
    question_id = d['question_id']
    id_ = f"MMLUPro-{question_id}"
    cnt += 1
    
    problem = d['question'] + "\n"
    options = d['options']
    answer = d['answer']
    
    mapping = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J"}
    for i, c in enumerate(options):
        problem += f"\n({mapping[i]}) {c}"
    problem += "\n"
    problem += "Please write your final answer in the form of \\boxed{" + f"a letter from A to {mapping[len(options) - 1]}" + "}."
    
    new_d = {
        "id": id_,
        "problem": problem,
        "gt_solution": answer,
        "gt_answer": answer,
    }
    
    our_data.append(new_d)
    
    category = d['category']
    if category in subfield_data:
        subfield_data[category].append(new_d) 

with open(tgt_file, "w") as tgt_f:
    for d in our_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
        
print(f"Saved {len(our_data)} examples to {tgt_file}")

for subfield in subfield_data:
    tgt_file = os.path.join(tgt_dir, f"MMLUPro{subfield2name[subfield]}.jsonl")
    with open(tgt_file, "w") as tgt_f:
        for d in subfield_data[subfield]:
            d_str = json.dumps(d)
            tgt_f.write(d_str + "\n")
    print(f"Saved {len(subfield_data[subfield])} examples to {tgt_file}")

no_math_data = []
for key, value in subfield_data.items():
    if key != "math":
        no_math_data.extend(value)

tgt_file = os.path.join(tgt_dir, "MMLUProNoMath.jsonl")
with open(tgt_file, "w") as tgt_f:
    for d in no_math_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
print(f"Saved {len(no_math_data)} examples to {tgt_file}")

STEM_data = []
for key, value in subfield_data.items():
    # MMLUProPhysics	MMLUProChemistry	MMLUProComputerScience	MMLUProEngineering	MMLUProBiology	MMLUProEconomics
    if key in ["physics", "chemistry", "computer science", "engineering", "biology", "economics"]:
        STEM_data.extend(value)

tgt_file = os.path.join(tgt_dir, "MMLUProNoMathSTEM.jsonl")
with open(tgt_file, "w") as tgt_f:
    for d in STEM_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
print(f"Saved {len(STEM_data)} examples to {tgt_file}")