import os, json


def load_data(file_path, file_type):
    if file_type == "jsonl":
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    elif file_type == "json":
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    else:
        raise NotImplementedError(f"File type {file_type} not supported")


def make_data_chunk(data, total_chunks, chunk_idx):
    indices = list(range(len(data)))
    chunk_size = (len(data) + total_chunks - 1) // total_chunks
    data = data[chunk_size * chunk_idx : chunk_size * (chunk_idx + 1)]
    indices = indices[chunk_size * chunk_idx : chunk_size * (chunk_idx + 1)]

    return data, indices


def split_into_batches(data, batch_size):
    """
    Splits a list of data into batches of a specified size.

    Parameters:
    - data (list): The list of items to be split into batches.
    - batch_size (int): The maximum size of each batch.

    Returns:
    - list: A list of batches (each a list) with a maximum length of batch_size.
    """
    assert batch_size > 0
    assert isinstance(data, list)
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
