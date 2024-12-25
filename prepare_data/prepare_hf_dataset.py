import json
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def prepare_hf_dataset(path_to_decomposition, orig_dataset_local_path, ds_out_path):
    """
    :param path_to_decomposition: output of decompose_tasks.py
    :param orig_dataset_local_path: output of heuristic_filtering.py
    :param ds_out_path: local path to save the ds at
    """
    with open(path_to_decomposition, 'rt') as f:
        tasks = json.load(f)
    orig_data = load_from_disk(orig_dataset_local_path)
    print("getting ids from original dataset")
    tasks_to_ids = {sample["task"].strip(): sample["conversation_id"] for sample in tqdm(orig_data)}
    rows_in_df = []
    count_0 = 0
    count_1 = 0
    for task in tasks:
        decomposition = tasks[task]
        if len(decomposition) == 0:
            count_0 += 1
            continue
        elif len(decomposition) == 1:
            count_1 += 1
        row_in_df = {"task": task.strip(),
                     "conversation_id": tasks_to_ids[task.strip()],
                     "decomposition": tasks[task]}
        rows_in_df.append(row_in_df)
    df = pd.DataFrame(rows_in_df)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, validation_df = train_test_split(train_df, test_size=0.125, random_state=42)
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(validation_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    ds = DatasetDict()

    ds['train'] = train_ds
    ds['validation'] = val_ds
    ds['test'] = test_ds

    print(f"saving the dataset locally to {ds_out_path}")
    ds.save_to_disk(ds_out_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--decomposition")
    parser.add_argument("--orig_dataset")
    parser.add_argument("--ds_out_path")

    args = parser.parse_args()
    prepare_hf_dataset(args.decomposition, args.orig_dataset, args.ds_out_path)
