import os
from datasets import load_dataset
from argparse import ArgumentParser


def leave_only_first_request(example):
    example["conversation"] = example["conversation"][0]["content"]
    return example

def filter_data(dataset):
    updated_dataset = dataset.map(leave_only_first_request)
    rename_col = updated_dataset.rename_column("conversation", "task")
    no_code = rename_col.filter(lambda example: "code" not in example["task"].lower() and example["language"] == "English")
    return no_code

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--out_dir")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset(args.dataset)
    dataset_name = args.dataset.split('/')[-1]

    for split in ds:
        print(f"FILTERING {args.dataset}---{split}")
        filtered_ds = filter_data(ds[split])
        out_path = os.path.join(args.out_dir, f"{dataset_name}-heuristic-filtered-{split}")
        filtered_ds.save_to_disk(out_path)
        print(f"saved at {out_path}")

