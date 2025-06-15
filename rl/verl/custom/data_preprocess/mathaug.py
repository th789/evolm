import os
import datasets
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_src_dir', default='/path/to/rl')
    parser.add_argument('--dataset_name', default='mathaug')
    args = parser.parse_args()
    
    local_dir = os.path.join(args.data_src_dir, args.dataset_name)
    os.makedirs(local_dir, exist_ok=True)
    
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            problem = example.pop('problem')
            gt_solution = example.pop('gt_solution')
            gt_answer = extract_solution(gt_solution)
            data = {
                "data_source": "ZhentingNLP/mathaug-disjoint",
                "prompt": [{
                    "role": "user",
                    "content": problem
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": gt_answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    data_source = 'ZhentingNLP/mathaug-disjoint'
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    
    test_dataset = dataset['test']
    test_dataset = test_dataset.shuffle(seed=42).select(range(1000))
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    for key in dataset.keys():
        if key != "test" and "first" not in key:
            train_dataset = dataset[key]
            train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
            train_dataset.to_parquet(os.path.join(local_dir, f'{key}.parquet'))
