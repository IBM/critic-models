from argparse import ArgumentParser

from openai import OpenAI
from datasets import load_dataset
import os
from utils.filter_data_consts import DECOMPOSE_PROMPT, filter_answer
from tqdm import tqdm
import json

ds_name = "gililior/arena_raw"

def decompose_with_gpt4o(out_json, max_len):
    client = OpenAI()
    ds = load_dataset(ds_name, split="all")
    existing_json = {}
    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            existing_json = json.load(f)
        print(f"loaded existing json, {len(existing_json)} samples\n")
    for row in tqdm(ds):
        task = row["task"]
        id_conv = row["conversation_id"]
        if id_conv in existing_json:
            continue
        message = DECOMPOSE_PROMPT.format(instruction=task)
        completion = client.chat.completions.create(
            messages=[{'role': 'user', 'content': message}],
            model="gpt-4o-2024-08-06",
            temperature=1,
            max_tokens=500
        )
        answer = completion.choices[0].message.content
        filtered_answer = filter_answer(answer)
        existing_json[id_conv] = {"conversation_id": id_conv, "task": task, "raw_answer": answer,
                                  "gpt4_constraints": filtered_answer, "llama3.1-8b_constraints": row["decomposition"]}
        if len(existing_json) >= 1:
            break
        if len(existing_json) % 20 == 0:
            print(f"saving another 20 tasks")
            with open(out_json, "w") as f:
                json.dump(existing_json, f)
        if len(existing_json) >= max_len > 0:
            break
    with open(out_json, "w") as f:
        json.dump(existing_json, f)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--out_json", required=True,
                        help="path to ds to save predictions to")
    parser.add_argument("--max_len", type=int, default=-1)
    args = parser.parse_args()
    decompose_with_gpt4o(args.out_json, args.max_len)
