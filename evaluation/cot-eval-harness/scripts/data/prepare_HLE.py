import os, json


tgt_file = "data/cooked/test/HLE.jsonl"

tgt_dir = os.path.dirname(tgt_file)
os.makedirs(tgt_dir, exist_ok=True)


def cnt_tokens(s):
    return len(s.split()) * 1.3


from datasets import load_dataset

ds = load_dataset("cais/hle", split="test", streaming=True)

our_data = []
cnt = 0
for d in ds:
    unique_id = d['id']
    id_ = f"HLE-{unique_id}"
    cnt += 1
    
    question = d['question']
    
    if cnt_tokens(question) > 1024:
        continue
    
    image = d['image']
    answer = d['answer']
    
    if len(image) > 0:
        continue

    new_d = {
        "id": id_,
        "problem": question,
        "gt_solution": answer,
        "gt_answer": answer
    }
    
    our_data.append(new_d)

with open(tgt_file, "w") as tgt_f:
    for d in our_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
        
print(f"Saved {len(our_data)} examples to {tgt_file}")