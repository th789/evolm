import os
import pandas as pd
import json
from transformers import AutoTokenizer

from pathlib import Path
import orjson
from tqdm import tqdm

import argparse
from datetime import datetime


#calculate number of files in a folder
def count_files_in_directory(directory_path):
    """
    Count the number of files in a given directory.
    
    Parameters:
    directory_path (str): Path to the directory
    
    Returns:
    int: Number of files in the directory
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Path {directory_path} is not a valid directory")
    
    file_count = 0
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            file_count += 1
    
    return file_count



loads = orjson.loads
dumps = orjson.dumps

def process_one_json_file(input_path, output_path, tokenizer):
    """
    Process a single JSONL file and compute the total number of tokens in this json file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        tokenizer: tokenizer used to calculate number of tokens, AutoTokenizer
    """
    num_tokens_in_json_file = 0

    with open(input_path, "rb", buffering=1024*1024) as fin, \
            open(output_path, "wb", buffering=1024*1024) as fout:
        #bind tokenizer function once
        get_input_tokens = tokenizer.encode

        #load json file line by line
        for i, line in enumerate(fin, 1):
            data = loads(line) #load one doc in json file
            num_tokens_in_doc = len(get_input_tokens(data["text"])) #compute number of tokens
            data["token_count_llama2"] = num_tokens_in_doc #add number of tokens to dictionary
            num_tokens_in_json_file += num_tokens_in_doc #add number of tokens to running total
            fout.write(dumps(data) + b"\n") #write new json file line by line

    return num_tokens_in_json_file



def main():
    """
    Main function to count the number of tokens in a domain.
    """

    start_time = datetime.now()
    print('Start time: ', start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print('')
    
    from private.access import storage_dir

    #parse arguments
    parser = argparse.ArgumentParser(description="Count the number of tokens in a domain.")
    parser.add_argument(
        "--domain", type=str, required=True, 
        help="Domain in FineFineWeb for which to count the number of llama2 tokens"
        )
    parser.add_argument(
        "--storage_dir", type=str, 
        help=(
            "Directory where FineFineWeb sample files are stored + where output will be saved. "
            "Note: {storage_dir}/finefineweb-sample/{domain} should exist. "
            "{storage_dir}/ffw_counted/{domain} will be created."
        ),
        default=storage_dir
        )
    parser.add_argument(
        "--model_dir", type=str, 
        help="Directory of model which contains Llama2 tokenizer",
        default="evolm/models/hf_ckpts/meta-llama/Llama-2-7b-hf"
        )
    args = parser.parse_args()

    
    domain = args.domain
    storage_dir = args.storage_dir
    model_dir = args.model_dir


    #create folder to store output
    output_dir = Path(f"{storage_dir}/ffw_counted/{domain}")
    output_dir.mkdir(parents=True, exist_ok=True)

    #count number of json files in this domain's folder
    print(f'Counting number of json files for domain...')
    num_json_files_for_domain = count_files_in_directory(f"{storage_dir}/finefineweb-sample/{domain}")

    #load tokenizer
    print(f"Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    #process each json file
    print(f'Processing {num_json_files_for_domain} json files for domain...')
    num_tokens_in_domain = 0 #running counter for total number of tokens in domain

    for i in tqdm(range(num_json_files_for_domain)):
        file_num = str(i).zfill(6)
        file_path = f"{storage_dir}/finefineweb-sample/{domain}/{domain}_{file_num}.jsonl"
        output_path = f"{storage_dir}/ffw_counted/{domain}/{domain}_{file_num}.jsonl"

        num_tokens_in_json_file = process_one_json_file(file_path, output_path, tokenizer)
        num_tokens_in_domain += num_tokens_in_json_file
        print(f"Running total number of tokens in domain: {num_tokens_in_domain}")

    ###count number of documents for domain
    #count lines in last jsonl file
    last_file_num = str(num_json_files_for_domain-1).zfill(6)
    last_json_file_path = f"{storage_dir}/finefineweb-sample/{domain}/{domain}_{last_file_num}.jsonl"
    with open(last_json_file_path, "r", encoding="utf-8") as f:
        n_lines_last_file = sum(1 for _ in f)
    #calculate total number of docs
    num_docs_for_domain = (num_json_files_for_domain-1)*100000 + n_lines_last_file

    ###save info about this domain: number of tokens and number of docs
    file_path = os.path.abspath(__file__) #path of current file
    current_dir = os.path.dirname(file_path) #directory of current file
    domain_info_file = f"{current_dir}/ffw_sample_domain_info.json"
    #write info to file
    with open(domain_info_file, "a", encoding="utf-8") as f:
        domain_info = {
            'domain': domain, 
            'n_json_files': num_json_files_for_domain,
            'n_docs': num_docs_for_domain, 
            'n_tokens_llama2': num_tokens_in_domain}
        f.write(json.dumps(domain_info, ensure_ascii=False) + "\n")

    print('')
    print(f'Domain: {domain}')
    print(f'Number of json files in domain: {num_json_files_for_domain}')
    print(f'Number individual documents in domain: {num_docs_for_domain}')
    print(f'Number Llama2 token in domain: {num_tokens_in_domain}')

    end_time = datetime.now()
    print('\nStart time: ', start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print('End time: ', end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print('Run time: ', end_time - start_time)
    print('Complete!')


    

if __name__ == "__main__":
    main()