import os
import json
import numpy as np
import pandas as pd

dir_name = "/Users/gililior/research/py_repos/WildIFEval/llm_aaj_scores"
prefix_0shot = "llm-aaj-llama3.1-70b.{model_name}-0shot-wild-if-eval.json"

dir_revision_scores = "/Users/gililior/research/datasets/wild-if-eval-revisions/llm_aaj/7-9b-models"
prefix_revision = "decomposition-evaluation-llama3.1-70b.{revision_model}-revise-one-step-{generator_model}.json"


def parse_scores(scores, tasks_list):
    mean_scores = []
    for sample in tasks_list:
        task_scores = scores[sample]["scores"]
        binary_scoring = []
        for score in task_scores.values():
            if score == "ERR" or score < 0.5:
                binary_scoring.append(0)
            else:
                binary_scoring.append(1)
        mean_constraint_scores = np.mean(binary_scoring)
        mean_scores.append(mean_constraint_scores)
    return mean_scores

if __name__ == '__main__':
    zero_shot_paths = {model: os.path.join(dir_name, prefix_0shot).format(model_name=model) for model in ["gemma-2-9b", "Llama-3.1-8B", "llama3.3-70b"]}
    revision_paths = {(rev_model, gen_model): os.path.join(dir_revision_scores, prefix_revision).format(revision_model=rev_model, generator_model=gen_model) for rev_model in ["gemma-2-9b", "Llama-3.1-8B", "Qwen2.5-7b"] for gen_model in ["gemma-2-9b", "Llama-3.1-8B", "Qwen2.5-7b"]}

    random_path_from_dict = list(revision_paths.values())[0]
    with open(random_path_from_dict, "r") as f:
        random_scores = json.load(f)
        tasks_list = list(random_scores.keys())
    tasks_list.sort()

    zero_shot_scores = {}
    for model in zero_shot_paths:
        with open(zero_shot_paths[model], "r") as f:
            json_scores = json.load(f)
        zero_shot_scores[model] = parse_scores(json_scores, tasks_list)
    revision_scores = {}
    for models in revision_paths:
        with open(revision_paths[models], "r") as f:
            json_scores = json.load(f)
        revision_scores[models] = parse_scores(json_scores, tasks_list)

    pred_dir = "_output/modernbert_predictions"
    # pred_dir = "_output/simple_classifiers"

    all_predictions_paths = os.listdir(pred_dir)
    all_predictions_paths = [path for path in all_predictions_paths if path.endswith(".json")]
    all_predictions_paths.sort()

    # calculate_baselines
    with open(f"{pred_dir}/{all_predictions_paths[0]}", "r") as f:
        predictions_example = json.load(f)
        all_tasks_in_test = list(predictions_example.keys())

    random_preds = {sample: np.random.randint(0, 3) for sample in all_tasks_in_test}
    id_to_label = {0: "both_revision_and_bigger_not_better", 1: "revision_with_gemma_is_best", 2: "inference_big_llama_0shot_is_best"}
    all_small_model_preds = []
    all_revision_model_preds = []
    all_big_llama_model_preds = []
    random_model_preds = []
    all_gold = []
    big_model_calls = {"random_preds": 0, "small_model_preds": 0, "revision_model_preds": 0,
                       "big_llama_model_preds": len(all_tasks_in_test), "gold": 0}
    revision_calls = {"random_preds": 0, "small_model_preds": 0, "revision_model_preds": len(all_tasks_in_test),
                      "big_llama_model_preds": 0, "gold": 0}
    for task in all_tasks_in_test:
        index_sample = tasks_list.index(task)
        all_small_model_preds.append(zero_shot_scores["Llama-3.1-8B"][index_sample])
        all_revision_model_preds.append(revision_scores[("gemma-2-9b", "Llama-3.1-8B")][index_sample])
        all_big_llama_model_preds.append(zero_shot_scores["llama3.3-70b"][index_sample])
        all_possible_labels = [all_small_model_preds[-1], all_revision_model_preds[-1], all_big_llama_model_preds[-1]]
        gold_index = np.argmax(all_possible_labels)
        if gold_index == 1:
            revision_calls["gold"] += 1
        elif gold_index == 2:
            big_model_calls["gold"] += 1
        all_gold.append(all_possible_labels[gold_index])
        random_int = random_preds[task]
        random_model_preds.append(all_possible_labels[random_int])
        if random_int == 1:
            revision_calls["random_preds"] += 1
        elif random_int == 2:
            big_model_calls["random_preds"] += 1
    print("random preds:", np.mean(random_model_preds).round(3))
    print("small model preds:", np.mean(all_small_model_preds).round(3))
    print("revision model preds:", np.mean(all_revision_model_preds).round(3))
    print("big llama model preds:", np.mean(all_big_llama_model_preds).round(3))
    print("gold:", np.mean(all_gold).round(3))

    all_baselines = {"random_preds": random_model_preds,
                     "small_model_preds": all_small_model_preds,
                     "revision_model_preds": all_revision_model_preds,
                     "big_llama_model_preds": all_big_llama_model_preds,
                     "gold": all_gold}

    all_preds = {}
    big_model_calls_preds = {}
    revision_calls_preds = {}
    for path in all_predictions_paths:
        with open(f"{pred_dir}/{path}", "r") as f:
            predictions = json.load(f)
        pred_scores = []
        key_name = path.replace(".json", "")
        big_model_calls_preds[key_name] = 0
        revision_calls_preds[key_name] = 0
        for task in all_tasks_in_test:
            pred = predictions[task]
            index_sample = tasks_list.index(task)
            if id_to_label[pred] == "both_revision_and_bigger_not_better":
                col = zero_shot_scores["Llama-3.1-8B"]
            elif id_to_label[pred] == "revision_with_gemma_is_best":
                col = revision_scores[("gemma-2-9b", "Llama-3.1-8B")]
                revision_calls_preds[key_name] += 1
            else:
                col = zero_shot_scores["llama3.3-70b"]
                big_model_calls_preds[key_name] += 1
            score = col[index_sample]
            pred_scores.append(score)
        all_preds[path.replace(".json", "")] = pred_scores

    results_df = pd.DataFrame()
    all_base_names = list(all_baselines.keys())
    all_pred_names = list(all_preds.keys())
    names = all_base_names + all_pred_names

    means = [np.mean(all_baselines[base]).round(3) for base in all_base_names]
    means.extend([np.mean(all_preds[pred]).round(3) for pred in all_pred_names])

    stds = [np.std(all_baselines[base]).round(3) for base in all_base_names]
    stds.extend([np.std(all_preds[pred]).round(3) for pred in all_pred_names])

    revision_calls_all = [revision_calls[base] for base in all_base_names]
    revision_calls_all.extend([revision_calls_preds[pred] for pred in all_pred_names])

    big_model_calls_all = [big_model_calls[base] for base in all_base_names]
    big_model_calls_all.extend([big_model_calls_preds[pred] for pred in all_pred_names])

    results_df["name"] = names
    results_df["mean"] = means
    results_df["std"] = stds
    results_df["revision_calls"] = revision_calls_all
    results_df["big_model_calls"] = big_model_calls_all
    results_df.to_csv("_output/results.csv", index=False)


    deltas = pd.DataFrame()
    deltas["llama3.1-8b-0shot"] = zero_shot_scores["Llama-3.1-8B"]
    deltas["0shot-gemma-minus-llama"] = np.array(zero_shot_scores["gemma-2-9b"]) - np.array(zero_shot_scores["Llama-3.1-8B"])
    deltas["0shot-llama-big-minus-llama-small"] = np.array(zero_shot_scores["llama3.3-70b"]) - np.array(zero_shot_scores["Llama-3.1-8B"])
    deltas["self-revision-llama-small-vs-0shot-llama-small"] = np.array(revision_scores[("Llama-3.1-8B", "Llama-3.1-8B")]) - np.array(zero_shot_scores["Llama-3.1-8B"])
    deltas["self-revision-llama-small-vs-0shot-llama-big"] = np.array(revision_scores[("Llama-3.1-8B", "Llama-3.1-8B")]) - np.array(zero_shot_scores["llama3.3-70b"])
    deltas["gemma-revise-llama-small-vs-0shot-llama-small"] = np.array(revision_scores[("gemma-2-9b", "Llama-3.1-8B")]) - np.array(zero_shot_scores["Llama-3.1-8B"])
    deltas["gemma-revise-llama-small-vs-0shot-llama-big"] = np.array(revision_scores[("gemma-2-9b", "Llama-3.1-8B")]) - np.array(zero_shot_scores["llama3.3-70b"])
    deltas["llama-small-revise-gemma-vs-0shot-llama-small"] = np.array(revision_scores[("Llama-3.1-8B", "gemma-2-9b")]) - np.array(zero_shot_scores["Llama-3.1-8B"])
    deltas["llama-small-revise-gemma-vs-0shot-llama-big"] = np.array(revision_scores[("Llama-3.1-8B", "gemma-2-9b")]) - np.array(zero_shot_scores["llama3.3-70b"])
    deltas.to_csv("_output/deltas.csv", index=False)


    samples_with_small_model_perfect_scoring = np.array(zero_shot_scores["Llama-3.1-8B"])==1
    gemma_revise_llama_doesnt_improve = np.array(revision_scores[("gemma-2-9b", "Llama-3.1-8B")]) <= np.array(zero_shot_scores["Llama-3.1-8B"])
    bigger_llama_no_better_than_small = np.array(zero_shot_scores["llama3.3-70b"]) <= np.array(zero_shot_scores["Llama-3.1-8B"])
    both_revision_and_bigger_not_better = np.logical_and(gemma_revise_llama_doesnt_improve, bigger_llama_no_better_than_small)

    revision_with_gemma_is_best = deltas["gemma-revise-llama-small-vs-0shot-llama-big"] >= 0
    revision_with_gemma_is_best = np.logical_and(revision_with_gemma_is_best, ~both_revision_and_bigger_not_better)

    inference_big_llama_0shot_is_best = deltas["gemma-revise-llama-small-vs-0shot-llama-big"] < 0
    inference_big_llama_0shot_is_best = np.logical_and(inference_big_llama_0shot_is_best, ~both_revision_and_bigger_not_better)

    dataset = pd.DataFrame()
    dataset["sample"] = tasks_list
    dataset["small_model_perfect"] = samples_with_small_model_perfect_scoring
    dataset["both_revision_and_bigger_not_better"] = both_revision_and_bigger_not_better
    dataset["revision_with_gemma_is_best"] = revision_with_gemma_is_best
    dataset["inference_big_llama_0shot_is_best"] = inference_big_llama_0shot_is_best
    dataset.to_csv("_output/dataset.csv", index=False)

    # prepare dataset of all small models refining llama3.1-8b
    qwen_revise_gemma = np.array(revision_scores[("Qwen2.5-7b", "gemma-2-9b")])
    llama_revise_gemma = np.array(revision_scores[("Llama-3.1-8B", "gemma-2-9b")])
    gemma_revise_gemma = np.array(revision_scores[("gemma-2-9b", "gemma-2-9b")])
    gemma_0shot = np.array(zero_shot_scores["gemma-2-9b"])
    dataset = pd.DataFrame()
    dataset["sample"] = tasks_list
    dataset["gemma_0shot"] = gemma_0shot
    dataset["qwen_revise_gemma"] = qwen_revise_gemma
    dataset["llama_revise_gemma"] = llama_revise_gemma
    dataset["gemma_revise_gemma"] = gemma_revise_gemma
    best_revision_model = []
    all_three_models_same = []
    for _, row in dataset.iterrows():
        revision_scores_current_row = np.array([row["qwen_revise_gemma"], row["llama_revise_gemma"], row["gemma_revise_gemma"]])
        best_rev_score = revision_scores_current_row.max()
        best_rev_models = revision_scores_current_row == best_rev_score
        # if more than one, choose randomly
        best_rev_model = np.random.choice(np.where(best_rev_models)[0])
        best_revision_model.append(best_rev_model)
        all_three_models_same.append(best_rev_models.sum() == 3)
    dataset["label"] = best_revision_model
    dataset["all_three_models_same"] = all_three_models_same

    dataset.to_csv("_output/dataset_routing_same_size_gemma_generator.csv", index=False)


    print("\npredictions (same size):")

    test_tasks = []
    predictions_dir = "_output/simple_classifiers_same_size"
    for path in os.listdir(predictions_dir):
        with open(f"{predictions_dir}/{path}", "r") as f:
            predictions = json.load(f)
        if len(test_tasks) == 0:
            test_tasks = list(predictions.keys())
        scores = []
        for _, row in dataset.iterrows():
            if row["sample"] not in test_tasks:
                continue
            pred = predictions[row["sample"]]
            all_possible_scores = [row["qwen_revise_gemma"], row["llama_revise_gemma"], row["gemma_revise_gemma"]]
            scores.append(all_possible_scores[pred])
        print(f"{path.replace('.json', '')}: {np.mean(scores)}")

    # same size baselines
    dataset_reduced = dataset[dataset["sample"].isin(test_tasks)]

    predict_all_qwen = np.mean(dataset_reduced["qwen_revise_gemma"])
    predict_all_llama = np.mean(dataset_reduced["llama_revise_gemma"])
    predict_all_gemma = np.mean(dataset_reduced["gemma_revise_gemma"])
    gold = []
    random = []
    for _, row in dataset_reduced.iterrows():
        all_possible_scores = [row["qwen_revise_gemma"], row["llama_revise_gemma"], row["gemma_revise_gemma"]]
        gold.append(all_possible_scores[row["label"]])
        random.append(all_possible_scores[np.random.randint(0, 3)])
    gold_mean = np.mean(gold)
    random_mean = np.mean(random)

    print("baselines:")
    print(f"predict_all_qwen: {predict_all_qwen}")
    print(f"predict_all_llama: {predict_all_llama}")
    print(f"predict_all_gemma: {predict_all_gemma}")
    print(f"gold: {gold_mean}")
    print(f"random: {random_mean}")


    print("\n\n")
    for col in dataset.columns[1:]:
        print(f"{col}: {dataset[col].sum()/len(dataset)}")




