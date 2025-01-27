import os.path
import numpy as np
import json
from itertools import product

DUMP = "train"

SCORES_DIR = "/Users/gililior/research/datasets/arena_data_v2/llm_aaj/{step}"
SCORES_PREFIX = "decomposition-evaluation-llama3.1-70b."
PATH = os.path.join(SCORES_DIR, SCORES_PREFIX)


def calculate(init_paths, revise_paths):
    all_jsons = {}
    all_tasks = set()
    for k in init_paths:
        with open(init_paths[k], 'r') as f:
            all_jsons[f"0-shot-{k}"] = json.load(f)
            all_tasks.update(all_jsons[f"0-shot-{k}"].keys())

    for (gen_model, revise_model) in revise_paths:
        if not os.path.exists(revise_paths[(gen_model, revise_model)]):
            continue
        with open(revise_paths[(gen_model, revise_model)], 'r') as f:
            all_jsons[f"{revise_model}-revise-{gen_model}"] = json.load(f)

    differences = {k: {r: [] for r in init_paths} for k in init_paths}

    all_scores = {k: [] for k in all_jsons}
    for task in all_tasks:
        for k in all_jsons:
            scores = list(all_jsons[k][task]["scores"].values())
            if len(scores) == 0:
                continue
            scores = [0 if s == 'ERR' else s for s in scores]
            mean_score = np.mean(scores)
            all_scores[k].append(mean_score)

    for gen_model in differences:
        for revise_model in differences[gen_model]:
            init_scores = all_scores[f"0-shot-{revise_model}"]

            if f"{revise_model}-revise-{gen_model}" not in all_scores:
                continue
            revise_scores = all_scores[f"{revise_model}-revise-{gen_model}"]
            differences[revise_model][gen_model] = np.array(revise_scores) - np.array(init_scores)

    mean_scores = {k: np.mean(all_scores[k]).round(3) for k in all_scores}
    for k in sorted(mean_scores, key=mean_scores.get):
        print(f"{k}: {mean_scores[k]}")

    print("\n\nDifference in scores:")
    for gen_model in differences:
        for revise_model in differences[gen_model]:
            if len(differences[revise_model][gen_model]) == 0:
                continue
            print(f"{revise_model} revise {gen_model} vs 0-shot {revise_model}: {np.mean(differences[revise_model][gen_model]).round(3)}")


if __name__ == '__main__':
    llama_models = ["Llama-3.1-8B-Instruct", "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"]
    gemma_models = ["gemma-2-2b-it", "gemma-2-9b-it"]

    for name, models in zip(["Llama", "Gemma"], [llama_models, gemma_models]):
        print(name)
        if name == 'Llama':
            to_remove = "-Instruct"
        else:  # Gemma
            to_remove = "-it"
        init = {model.replace(to_remove, ""): PATH.format(step="initial") + model + f"-{DUMP}-init-gen.json"
                for model in models}
        revise = {(generator.replace(to_remove, ""), revision_model.replace(to_remove, "")): PATH.format(
            step="revised") + f"{revision_model}-revise-one-step-{generator}-{DUMP}.json"
                  for (generator, revision_model) in product(models, models)}
        calculate(init, revise)

