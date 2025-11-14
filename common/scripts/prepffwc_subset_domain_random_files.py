import json
import pandas as pd
import random
import orjson
import os
from datetime import datetime

import pyarrow.json as paj
import pyarrow.parquet as pq
import argparse


def shuffle_indices(n, seed=42):
    random.seed(seed)
    indices = list(range(0, n))
    random.shuffle(indices)
    return indices



def calc_file_info(file_path):
    """
    Count the total number of tokens + total number of documents (# lines) in a jsonl file.
    """

    total_tokens = 0
    n_docs = 0

    with open(file_path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            doc = orjson.loads(line)
            total_tokens += doc["token_count_llama2"]
            n_docs += 1
    
    return total_tokens, n_docs



def main():
    
    start_time = datetime.now()
    print('Start time: ', start_time.strftime("%Y-%m-%d %H:%M:%S"))
    
    from private.access import storage_dir

    #parse args
    parser = argparse.ArgumentParser(description="Get subset of FFW-sample for a given domain.")
    parser.add_argument(
        "--domain", type=str, required=True, 
        help="Domain in FineFineWeb for which to obtain subset"
        )
    parser.add_argument(
        "--storage_dir", type=str, 
        help=(
            "Directory where FineFineWeb sample files with token counts are stored + where output will be saved. "
            "Note: {storage_dir}/ffw_counted/{domain}/{domain}_XXXXXX.jsonl files should exist. "
            "{storage_dir}/ffw_mysubset_20BT/{domain}_XXXXXX.parquet files will be created."
        ),
        default=storage_dir
        )
    args = parser.parse_args()

    domain = args.domain
    storage_dir = args.storage_dir


    #get number of tokens desired for this domain
    num_tokens_dict = json.load(open("evolm/common/scripts/ffw_sample_num_tokens_llama2_for_mysubset_20BT.json"))
    n_tokens_desired = num_tokens_dict[domain]

    print(f"\nDomain: {domain}")
    print("Desired number of tokens: ", n_tokens_desired)
    

    #create list of document indices with random order
    print('\nShuffling file indices...')
    domain_data = pd.read_json("evolm/common/scripts/ffw_sample_domain_info.json", lines=True)
    n_files_for_domain = domain_data[domain_data['domain'] == domain]['n_json_files'].item()
    file_idxs_shuffled = shuffle_indices(n_files_for_domain) #indices range from 0 to (n_files_for_domain-1)
    print(f"   # files for domain: {n_files_for_domain}")

    #set up directory to save output parquetfiles
    output_dir = f"{storage_dir}/ffw_mysubset_20BT"
    os.makedirs(output_dir, exist_ok=True)

    #intitalize tracking
    n_tokens_running_total = 0
    n_docs_for_domain_in_subset = 0
    chunk_counter = 0

    #for loop
    for i, file_idx in enumerate(file_idxs_shuffled):

        #print progress
        print(f"\nProcessing file domain_{file_idx:06d}.jsonl ({i+1} / {n_files_for_domain:,} total files)...")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Already processed {n_docs_for_domain_in_subset:,} documents")
        pct = (n_tokens_running_total / n_tokens_desired) * 100
        print(f"   Already obtained {n_tokens_running_total:,} / {n_tokens_desired:,} tokens, {pct:.1f} % complete")

        #check if this domain is complete
        if n_tokens_running_total >= n_tokens_desired:
            break

        #if this domain still needs tokens, use the whole file or iterate through file doc by doc
        if n_tokens_running_total < n_tokens_desired:

            #count #tokens + #documents in file
            file_path = f"{storage_dir}/ffw_counted/{domain}/{domain}_{file_idx:06d}.jsonl"
            n_tokens_in_file, n_docs_in_file = calc_file_info(file_path)

            #if keeping all the tokens in this file is <= n_tokens_desired, add all the documents in this file to the subset
            if n_tokens_running_total + n_tokens_in_file <= n_tokens_desired:
                print(f"   Using whole file (all {n_docs_in_file:,} documents, all {n_tokens_in_file:,} tokens)...")
                #save jsonl file as parquet file
                output_parquet_path = f"{output_dir}/{domain}_{chunk_counter:06d}.parquet"
                table = paj.read_json(file_path)
                pq.write_table(table, output_parquet_path)
                print(f"   Saved chunk (chunk {chunk_counter}) with {n_docs_in_file:,} documents/rows at {output_parquet_path}")
                #update counters
                chunk_counter += 1
                n_docs_for_domain_in_subset += n_docs_in_file
                n_tokens_running_total += n_tokens_in_file
                

            #else, go through each document in the file until n_tokens_desired is reached
            else:
                print(f"   Using part of file (< {n_docs_in_file:,} documents, < {n_tokens_in_file:,} tokens)...")
                data_to_save = []

                #load jsonl file document by document (line by line)
                with open(file_path, "rb") as f:
                    for line in f:
                        doc = orjson.loads(line)

                        #if under n_tokens_desired, add document to list
                        if n_tokens_running_total < n_tokens_desired:
                            data_to_save.append(doc)
                            n_docs_for_domain_in_subset += 1
                            n_tokens_running_total += doc['token_count_llama2']
                        else: #reached number of desired tokens
                            break
                
                #save collection of documents to parquet file
                df = pd.DataFrame(data_to_save)
                output_parquet_path = f"{output_dir}/{domain}_{chunk_counter:06d}.parquet"
                df.to_parquet(output_parquet_path, engine="pyarrow", index=False)
                print(f"   Saved last chunk (chunk {chunk_counter}) with {len(data_to_save):,} documents/rows at {output_parquet_path}")
                break


    #print final statistics
    print("")
    print(f"Domain: {domain}")
    print(f"Wanted {n_tokens_desired:,} tokens")
    print(f"Obtained {n_tokens_running_total:,} tokens")
    print(f"Saved {n_docs_for_domain_in_subset:,} documents in {chunk_counter+1:,} chunks at {output_dir}")


    #save info about this domain to json file
    file_path = os.path.abspath(__file__) #path of current file
    current_dir = os.path.dirname(file_path) #directory of current file
    domain_info_file = f"{current_dir}/ffw_domain_info_mysubset_20BT.json"
    with open(domain_info_file, "a", encoding="utf-8") as f:
        domain_info_subset = {
            'domain': domain,
            'n_json_files': chunk_counter+1, #since chunk_counter starts at 0 
            'n_docs': n_docs_for_domain_in_subset, 
            'n_tokens': n_tokens_running_total,
            'n_tokens_desired': n_tokens_desired,
            }
        f.write(json.dumps(domain_info_subset, ensure_ascii=False) + "\n")


    #print run time info
    end_time = datetime.now()
    print('\nStart time: ', start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print('End time: ', end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print('Run time: ', end_time - start_time, '(HH:MM:SS.microseconds)')
    print('Complete!')


if __name__ == "__main__":
    main()



