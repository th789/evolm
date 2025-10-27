print("Import packages...")
from datasets import load_dataset


print("Downloading MetaMathQA dataset...")
ds = load_dataset("meta-math/MetaMathQA", split="train")
print("Number of data points: ", len(ds))
print("Folder where dataset is stored: ", ds.cache_files)


print("Processing data into JSON format...")
# reformat dataset to contain json fields for alpaca format
alpaca_ds = ds.map(
    lambda x: {
        "instruction": x["query"],
        "input": "",
        "output": x["response"],
    },
    remove_columns=ds.column_names,  # drop original columns
)
# write dataset to a json file
out_path = "data/metamathqa.json"
alpaca_ds.to_json(out_path, orient="records", indent=4, lines=False, force_ascii=False)

print(f"Wrote {len(alpaca_ds):,} examples to {out_path}")


print("Complete!")