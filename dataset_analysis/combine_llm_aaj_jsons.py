import os

from datasets import load_dataset
import json
from argparse import ArgumentParser
from tqdm import tqdm
train_dir = "/Users/gililior/research/datasets/arena_data_v2/initial_generations"
rest_dir = "/Users/gililior/research/datasets/arena_data_v2/initial_generations_rest"


def main(out_dir):
    ds = load_dataset("gililior/wild-if-eval", split='test')
    for path in os.listdir(train_dir):
        print(path)
        if not path.endswith(".json"):
            continue
        train_path = os.path.join(train_dir, path)
        test_path = os.path.join(rest_dir, path.replace("train", "test"))
        if not os.path.exists(test_path):
            print(f"Test path {test_path} not found")
            test_path = test_path.replace("constested-lmsys-chat-1m", "temp-ds")
            if not os.path.exists(test_path):
                continue
        with open(train_path, "r") as f:
            train_json = json.load(f)
        if "predictions_key" in train_json:
            train_json = train_json[train_json["predictions_key"]]
        with open(test_path, "r") as f:
            test_json = json.load(f)
        if "predictions_key" in test_json:
            test_json = test_json[test_json["predictions_key"]]
        # combine train and test, but only for samples that appear in ds
        combined_json = {}

        for task in tqdm(ds["task"]):
            if task in train_json:
                combined_json[task] = train_json[task]
            elif task in test_json:
                combined_json[task] = test_json[task]
            else:
                print(f"Task {task} not found in either train or test json")
        out_path = os.path.join(out_dir, path.replace("train", "test"))
        with open(out_path, "w") as f:
            str_json = json.dumps(combined_json)
            f.write(str_json)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args.out_dir)
