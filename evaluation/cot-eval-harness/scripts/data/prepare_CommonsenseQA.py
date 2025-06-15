import os, json


tgt_file = "data/cooked/CommonsenseQA.jsonl"

tgt_dir = os.path.dirname(tgt_file)
os.makedirs(tgt_dir, exist_ok=True)

from datasets import load_dataset

ds = load_dataset("tau/commonsense_qa", split="validation", streaming=True)

our_data = []
cnt = 0
for d in ds:
    id_ = f"CommonsenseQA-{cnt:04d}"
    cnt += 1
    
    question = d['question']
    choices = d['choices']
    label = choices['label']
    assert label[0] == 'A'
    assert label[1] == 'B'
    assert label[2] == 'C'
    assert label[3] == 'D'
    assert label[4] == 'E'
    assert len(label) == 5
    text = choices['text']
    assert len(label) == len(text)
    answerKey = d['answerKey']
    assert answerKey in label, breakpoint()
    
    label2text = dict(zip(label, text))
    answerText = label2text[answerKey]
    
    problem = f"{question}\n\n"
    problem += f"Your final answer should be one of the following: "
    for t in text:
        assert "," not in t
        problem += f"{t}, "
    problem = problem[:-2] + "."
    
    new_d = {
        "id": id_,
        "problem": problem,
        "gt_solution": answerText,
        "gt_answer": answerText
    }
    
    our_data.append(new_d)

from random import shuffle
shuffle(our_data)

with open(tgt_file, "w") as tgt_f:
    for d in our_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
        
print(f"Saved {len(our_data)} examples to {tgt_file}")