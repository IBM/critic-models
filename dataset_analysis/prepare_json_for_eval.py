import json
import os

from datasets import load_dataset, concatenate_datasets

NUM_AGREEMENT = 50
NUM_UNIQUE = 50
NUM_PARTICIPANTS = 4

mapping = ['gili', 'liat', 'ariel', 'asaf']


def main():
    ds = load_dataset("gililior/wild-if-eval", split="test").shuffle(seed=42)
    agreement_samples = ds.select(range(NUM_AGREEMENT))
    for i in range(NUM_PARTICIPANTS):
        start = NUM_AGREEMENT + (i*NUM_UNIQUE)
        end = NUM_AGREEMENT + (i+1)*NUM_UNIQUE
        unique_samples = ds.select(range(start, end))
        combined_for_user = {line["task"]: line["decomposition"] for line in concatenate_datasets((agreement_samples, unique_samples))}
        with open(os.path.join("_output", f"task_decomposition_{mapping[i]}.json"), "w") as f:
            str_json = json.dumps(combined_for_user, indent=2)
            f.write(str_json)


if __name__ == '__main__':
    main()
