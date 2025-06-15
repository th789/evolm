import os, json


tgt_file = "data/cooked/test/TableBench.jsonl"

tgt_dir = os.path.dirname(tgt_file)
os.makedirs(tgt_dir, exist_ok=True)

from datasets import load_dataset

ds = load_dataset("Multilingual-Multimodal-NLP/TableBench", split="test", streaming=True)


def json_to_markdown_table(json_str):
    # Parse the JSON string into a dictionary
    table_data = json.loads(json_str)
    
    # Extract the columns and data
    columns = table_data["columns"]
    data = table_data["data"]

    # Assertions to ensure data validity
    assert isinstance(columns, list), "Columns should be a list."
    assert all(isinstance(col, str) for col in columns), "All columns should be strings."
    assert isinstance(data, list), "Data should be a list."
    assert all(isinstance(row, list) for row in data), "Each row in data should be a list."
    assert all(len(row) == len(columns) for row in data), "Each row should have the same number of elements as there are columns."
    
    # Start building the Markdown table
    markdown = "| " + " | ".join(columns) + " |\n"
    markdown += "|" + "|".join([" --- " for _ in columns]) + "|\n"
    
    # Add each row of data to the Markdown table
    for row in data:
        markdown += "| " + " | ".join(map(str, row)) + " |\n"
    
    return markdown


def count_tokens(s):
    s = s.split()
    num_tokens = 1.3 * len(s)
    return num_tokens


our_data = []
unique_problems = set()
cnt = 0
for d in ds:
    qtype = d['qtype']
    if qtype in ["Visualization", "DataAnalysis"]:
        continue
    instruction_type = d["instruction_type"]
    if instruction_type != "DP":
        continue
    
    unique_id = d["id"]
    id_ = f"TableBench-{unique_id}"
    cnt += 1
    table = d['table']
    question = d['question']
    answer = d['answer']
    table_md = json_to_markdown_table(table)
    
    problem = "You are given a table with the following data:\n\n"
    problem += f"{table_md}\n\n"
    problem += "Based on the table, answer the following question. If your answer is extracted from the table, make sure that the answer is exactly the same as the corresponding content in the table.\n\n"
    problem += f"{question}"
    
    if problem in unique_problems:
        continue
    unique_problems.add(problem)
    
    new_d = {
        "id": id_,
        "problem": problem,
        "gt_solution": answer,
        "gt_answer": answer
    }
    
    our_data.append(new_d)

our_data = [d for d in our_data if count_tokens(d["problem"]) < 3072]

with open(tgt_file, "w") as tgt_f:
    for d in our_data:
        d_str = json.dumps(d)
        tgt_f.write(d_str + "\n")
        
print(f"Saved {len(our_data)} examples to {tgt_file}")
