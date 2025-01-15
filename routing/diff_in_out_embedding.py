
import json

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_all_embeddings(config):
    all_models_sorted = sorted(config["generator_models"])
    all_embeddings = []
    tasks_ordered = None
    for model in all_models_sorted:
        embedding_output = config["representations_path"].format(generator_model=model, split="train",
                                                                 embedding_dir=config["embedding_dir"],
                                                                 representation="outputs")
        all_embeddings.append(np.load(embedding_output))
        tasks_path = embedding_output.replace("outputs.npy", "sorted_tasks.json")
        with open(tasks_path, 'r') as f:
            tasks = json.load(f)
        if tasks_ordered is None:
            tasks_ordered = tasks
        else:
            assert tasks_ordered == tasks

    return all_embeddings, tasks_ordered


def calc_differences(all_embeddings, tasks_ordered, config):
    sorted_models = sorted(config["generator_models"])
    all_rows = []
    for model_index in tqdm(range(len(all_embeddings))):
        init_path = config["init_path"].format(generator_model=sorted_models[model_index],
                                               split="train",
                                               init_dir=config["init_dir"],
                                               eval_prefix=config["evals_prefix"])
        init_scores = get_mean_scores(init_path, tasks_ordered)
        all_diffs_embeddings_for_gen_model = []
        all_diffs_scores_for_gen_model = []
        for other_model_index in range(len(all_embeddings)):
            diff_in_embedding = np.linalg.norm(all_embeddings[model_index] - all_embeddings[other_model_index], axis=1)
            all_diffs_embeddings_for_gen_model.append(diff_in_embedding)
            revised_path = config["revised_path"].format(generator_model=sorted_models[model_index],
                                                         split="train",
                                                         revised_dir=config["revised_dir"],
                                                         eval_prefix=config["evals_prefix"],
                                                         critic_model=sorted_models[other_model_index])
            revised_scores = get_mean_scores(revised_path, tasks_ordered)
            diff_in_scores = np.array(revised_scores) - np.array(init_scores)
            all_diffs_scores_for_gen_model.append(diff_in_scores)
        all_diffs_embeddings_for_gen_model = np.array(all_diffs_embeddings_for_gen_model)
        all_diffs_scores_for_gen_model = np.array(all_diffs_scores_for_gen_model)
        max_diffs_embeddings = np.max(all_diffs_embeddings_for_gen_model, axis=0)
        max_diffs_scores = np.max(all_diffs_scores_for_gen_model, axis=0)
        max_diff_corr = []
        max_diff_corr_not_self = []
        max_diff_corr_actually_improved = []
        actually_improved_and_not_self = []
        for i, task in enumerate(tasks_ordered):
            max_diff_embed_models = np.where(all_diffs_embeddings_for_gen_model[:, i] == max_diffs_embeddings[i])[0]
            max_diff_scores_models = np.where(all_diffs_scores_for_gen_model[:, i] == max_diffs_scores[i])[0]
            # has intersection between max_diff_embed_models and max_diff_scores_models
            if len(set(max_diff_embed_models).intersection(set(max_diff_scores_models))) > 0:
                max_diff_corr.append(1)
            else:
                max_diff_corr.append(0)
            if not np.all(max_diff_scores_models == model_index):
                max_diff_corr_not_self.append(max_diff_corr[-1])
                if max_diffs_scores[i] > 0:
                    actually_improved_and_not_self.append(max_diff_corr[-1])
            if max_diffs_scores[i] > 0:
                max_diff_corr_actually_improved.append(max_diff_corr[-1])
        all_rows.append({"gen_model": sorted_models[model_index],
                         "corr": np.mean(max_diff_corr),
                         "num_tasks": len(max_diff_corr),
                         "corr_not_self": np.mean(max_diff_corr_not_self),
                         "num_not_self": len(max_diff_corr_not_self),
                         "corr_actually_improved": np.mean(max_diff_corr_actually_improved),
                         "num_actually_improved": len(max_diff_corr_actually_improved),
                         "corr_actually_improved_and_not_self": np.mean(actually_improved_and_not_self),
                         "num_actually_improved_and_not_self": len(actually_improved_and_not_self),}
                        )
    results_df = pd.DataFrame(all_rows)
    return results_df


def get_mean_scores(path, tasks_ordered):
    with open(path, 'r') as f:
        init = json.load(f)
    all_scores = [list(init[t]["scores"].values()) for t in tasks_ordered]
    all_scores = [[0 if s == 'ERR' else s for s in scores] for scores in all_scores]
    all_scores = [np.mean(scores) for scores in all_scores]
    return all_scores


def main(config, out_path):
    all_embeddings, tasks_ordered = load_all_embeddings(config)
    results_df = calc_differences(all_embeddings, tasks_ordered, config)
    results_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    config_path = "utils/config_for_routing_local.json"
    output_path = "_output/corr_to_embeddings.csv"
    with open(config_path, 'r') as f:
        conf = json.load(f)
    main(conf, output_path)
