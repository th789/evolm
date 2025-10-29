print("Import packages...")
from datasets import load_dataset
import json

### Download data
print("Downloading MetaMathQA dataset...")
ds = load_dataset("meta-math/MetaMathQA", split="train")
print("Number of data points: ", len(ds))
print("Folder where dataset is stored: ", ds.cache_files)


# reformat dataset to contain json fields for alpaca format
print("Reformat dataset to contain json fields for alpaca format...")
alpaca_ds = ds.map(
    lambda x: {
        "instruction": x["query"],
        "input": "",
        "output": x["response"],
    },
    remove_columns=ds.column_names,  # drop original columns
)


#write json file manually to ensure proper format -- single JSON list
#alpaca_ds.to_json() writes several lists of dictionaries, want one list only
print("Writing final JSON file...")
out_path = "data/metamathqa.json"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("[\n")  # start JSON array
    for i, row in enumerate(alpaca_ds):
        json.dump(dict(row), f, ensure_ascii=False, indent=4)
        if i != len(ds) - 1:  # add comma between items, except the last
            f.write(",\n")
    f.write("\n]")  # end JSON array

print(f"Wrote {len(alpaca_ds):,} examples to {out_path}")


#OLD -- v2
# ### Process data into desired JSON format
# print("Processing data into JSON format...")

# # Export dataset to JSON lines in memory
# json_lines = ds.to_json(orient="records", lines=True)

# # Each line is a JSON object, so load them all into a single list
# records = [json.loads(line) for line in json_lines.splitlines()]

# # Write a proper single JSON list to file
# print("Writing final JSON file...")
# out_path = "data/metamathqa.json"
# with open(out_path, "w", encoding="utf-8") as f:
#     json.dump(records, f, indent=4, ensure_ascii=False)

# print(f"Wrote {len(records):,} examples to {out_path}")


#OLD -- v1
# # reformat dataset to contain json fields for alpaca format
# alpaca_ds = ds.map(
#     lambda x: {
#         "instruction": x["query"],
#         "input": "",
#         "output": x["response"],
#     },
#     remove_columns=ds.column_names,  # drop original columns
# )
# # write dataset to a json file
# out_path = "data/metamathqa.json"
# alpaca_ds.to_json(out_path, orient="records", indent=4, lines=False, force_ascii=False)

# print(f"Wrote {len(alpaca_ds):,} examples to {out_path}")


print("Complete!")