import json
import argparse
from datasets import load_dataset
import re
import multiprocessing
import os
from mbpp_data.run_tests import extract_code_from_output

def main(predictions_dir):
    formatted_dir = os.path.join(predictions_dir, "formatted")
    os.makedirs(formatted_dir, exist_ok=True)

    # Load predictions from the given JSON file
    for fname in os.listdir(predictions_dir):
        if ".json" not in fname:
            continue
        predictions_path = os.path.join(predictions_dir, fname)
        with open(predictions_path, "r") as f:
            predictions = json.load(f)
        if predictions.get("predictions_key") is not None:
            predictions = predictions[predictions.get("predictions_key")]
        output_list = []
        for task in predictions:
            only_code = extract_code_from_output(predictions[task][-1]["content"])
            out = f'"""\n{task}\n"""\n{only_code}'
            output_list.append([out])
        formatted_out_path = os.path.join(formatted_dir, fname)
        with open(formatted_out_path, "w") as f:
            json.dump(output_list, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=str, help="Path to the predictions JSON file.")
    args = parser.parse_args()

    main(args.predictions_dir)
