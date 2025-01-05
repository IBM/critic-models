import os.path
from argparse import ArgumentParser
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from itertools import product
import numpy as np
import json
from datasets import Dataset
import datasets
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier, KNeighborsRegressor
import pandas as pd
import scipy.stats as stats
from scipy.stats import entropy
from utils.analysis import prepare_df
# from routing.cluster_by_best_critic import parse_improvement_pairs
# from routing.pipeline import calc_scores
from collections import Counter
from tqdm import tqdm

NUM_KS = 10
NUM_ROUNDS = 1

IMPROVEMENT_THRESHOLD = 1

def generate_label(num_total_models, generator_index, critic_index):
    num_total_models += 1 # for the option of no critic model at all
    return generator_index*num_total_models + critic_index

def get_models_from_combined_label(label, num_models):
    num_models += 1 # for the option of no critic model at all
    critic_index = label%num_models
    generator_index = label//num_models
    return generator_index, critic_index

def get_all_scores(data):

    data["initial_scores_normalized"] = data["initial_score_sum"]/data["num_constraints"]
    data["revised_scores_normalized"] = data["revised_score_sum"] / data["num_constraints"]
    group_by_task = data.groupby("task")
    max_initial = group_by_task["initial_score_sum"].max()
    max_revised = group_by_task["revised_score_sum"].max()
    best_generator_model_on_train = data.groupby("generator_model")["initial_score_sum"].sum().idxmax()
    best_critic_model_on_train = data.groupby("critic_model")["revised_score_sum"].sum().idxmax()
    all_models_sorted = sorted(data["generator_model"].unique().tolist())
    num_models = len(all_models_sorted)

    by_task_dict = {}
    for task in tqdm(group_by_task.groups):
        only_task_df = data.loc[group_by_task.groups[task]]
        current_task_by_gen_model = only_task_df.groupby("generator_model")
        best_generator_for_current_task = current_task_by_gen_model["initial_score_sum"].max().idxmax()
        # critic_can_help_current_task_best_generator = current_task_by_gen_model["revised_score_sum"].max()[best_generator_for_current_task] > current_task_by_gen_model["initial_score_sum"].max()[best_generator_for_current_task]
        # critic_routing_helps_for_task = max_revised[task] > max_initial[task]
        # self_critic_helps_for_best_train_generator_df = only_task_df.loc[np.all(only_task_df[["critic_model", "generator_model"]] == [best_generator_model_on_train, best_generator_model_on_train], axis=1)]
        # self_critic_helps_for_best_train_generator = np.any(self_critic_helps_for_best_train_generator_df["initial_score_sum"] < self_critic_helps_for_best_train_generator_df["revised_score_sum"])
        # critic_routing_helps_for_best_train_generator_df =  only_task_df.loc[only_task_df["generator_model"]==best_generator_model_on_train]
        # critic_routing_helps_for_best_train_generator = np.any(critic_routing_helps_for_best_train_generator_df["initial_score_sum"] < critic_routing_helps_for_best_train_generator_df["revised_score_sum"])
        initial_scores_for_task = np.zeros((len(all_models_sorted),))
        revised_scores_for_task = np.zeros((len(all_models_sorted),len(all_models_sorted)))
        for i, gen_model in enumerate(all_models_sorted):
            initial_scores_for_task[i] = only_task_df.loc[only_task_df["generator_model"] == gen_model]["initial_scores_normalized"].max()
            for j, critic_model in enumerate(all_models_sorted):
                revised_scores_for_task[i, j] = only_task_df.loc[only_task_df["generator_model"] == gen_model].loc[
                    only_task_df["critic_model"] == critic_model]["revised_scores_normalized"].values[0]

        best_on_train_index = all_models_sorted.index(best_generator_model_on_train)
        best_critic_on_train_index = all_models_sorted.index(best_critic_model_on_train)
        best_on_current_task_index = all_models_sorted.index(best_generator_for_current_task)
        combined_scores = np.hstack((revised_scores_for_task, initial_scores_for_task.reshape(-1, 1)))
        max_score_each_row = np.max(combined_scores, axis=1)
        combined_critic_routing_labels = (combined_scores == max_score_each_row.reshape(-1, 1)).astype(int)
        critic_can_improve_labels = np.any(revised_scores_for_task > initial_scores_for_task.reshape(-1, 1), axis=1).astype(int)
        combined_label_gen_and_critic = np.where(np.max(combined_scores) == combined_scores)
        combined_label_gen_and_critic = generate_label(num_models, combined_label_gen_and_critic[0][0], combined_label_gen_and_critic[1][0])
        two_steps_label = generate_label(num_models, best_on_current_task_index, revised_scores_for_task[best_on_current_task_index].argmax())
        best_on_train_gen_and_critic = generate_label(num_models, best_on_train_index, best_critic_on_train_index)

        best_on_train_label = np.zeros((len(all_models_sorted)+1,))

        best_on_train_critic_scores = np.zeros((len(all_models_sorted)+1,))
        best_on_train_critic_scores[:-1] = revised_scores_for_task[best_on_train_index]
        best_on_train_critic_scores[-1] = initial_scores_for_task[best_on_train_index]
        best_on_train_label[best_on_train_critic_scores==best_on_train_critic_scores.max()] = 1
        max_init_score = initial_scores_for_task.max()
        by_task_dict[task] = {
            "task": task,
            "combined_scores": combined_scores,
            "critic_routing_labels": combined_critic_routing_labels,
            "critic_can_improve_labels": critic_can_improve_labels,
            "combined_label_gen_and_critic": combined_label_gen_and_critic,
            "two_steps_label": two_steps_label,
            "best_on_train_gen_and_critic": best_on_train_gen_and_critic,
            "model_routing_score": max_init_score,
            "critic_routing_on_best_train_score": best_on_train_critic_scores.max(),
            "best_on_train_label": best_on_train_label,
            "best_on_train_critic_scores": best_on_train_critic_scores,
            "best_on_train_upper_bound_score": best_on_train_critic_scores.max(),
            "best_on_train_used_critic": best_on_train_label[-1] == 0, # initial score is in the -1 index
            "self_critic_for_best_on_train_score": max(best_on_train_critic_scores[[best_on_train_index, -1]]),
            "initial_score_best_on_train": best_on_train_critic_scores[-1],
            "max_revised_score": revised_scores_for_task.max(),
            "gen_and_critic_routing_max_score": max([revised_scores_for_task.max(), max_init_score]),
            "critic_routing_on_model_routing": max([revised_scores_for_task[best_on_current_task_index].max(), max_init_score]),
            "self_critic_on_model_routing": max([revised_scores_for_task[best_on_current_task_index, best_on_current_task_index], max_init_score]),
            "best_critic_on_train_for_best_on_train_generation": max(best_on_train_critic_scores[[best_critic_on_train_index, -1]]),
            "best_critic_on_train_for_model_routing":  max([revised_scores_for_task[best_on_current_task_index, best_critic_on_train_index], max_init_score]),
        }



            # "best_generator_for_current_task": best_generator_for_current_task,
            # "best_critic_for_current_task": all_models_sorted[revised_scores_for_task.sum(axis=0).argmax()],
            # "best_gen_on_train": best_generator_model_on_train,
            # "init_score_for_best_gen_current_task": initial_scores_for_task.max(),
            # "init_score_for_best_gen_on_train": initial_scores_for_task[best_on_train_index],
            # "critic_best_on_train_improving_best_on_train": revised_scores_for_task[best_on_train_index, best_critic_on_train_index],
            # "critic_best_on_train_improving_best_current_task": revised_scores_for_task[best_on_current_task_index, best_critic_on_train_index],
            # "best_revised_score_for_best_gen_on_train": revised_scores_for_task[best_on_train_index].max(),
            # "best_critic_for_gen_best_on_train": all_models_sorted[revised_scores_for_task[best_on_train_index].argmax()],
            # "best_revised_score_for_best_gen_current_task": revised_scores_for_task[best_on_current_task_index].max(),
            # "best_critic_for_gen_best_current_task": all_models_sorted[revised_scores_for_task[best_on_current_task_index].argmax()],
        #     "self_critic_for_best_on_train": revised_scores_for_task[best_on_train_index, best_on_train_index],
        #     "self_critic_for_best_current_task": revised_scores_for_task[best_on_current_task_index, best_on_current_task_index],
        # }
        # by_task_dict[task].update({
        #     "self_critic_helps_best_on_train": by_task_dict[task]["self_critic_for_best_on_train"] > by_task_dict[task]["init_score_for_best_gen_on_train"],
        #     "critic_routing_helps_best_on_train": by_task_dict[task]["best_revised_score_for_best_gen_on_train"] > by_task_dict[task]["init_score_for_best_gen_on_train"],
        #     "self_critic_helps_best_current_task": by_task_dict[task]["self_critic_for_best_current_task"] >= by_task_dict[task]["best_revised_score_for_best_gen_current_task"],
        #     "critic_routing_helps_best_current_task": by_task_dict[task]["best_revised_score_for_best_gen_current_task"] >= by_task_dict[task]["best_revised_score_for_best_gen_current_task"],
        #     "max_revised_score": revised_scores_for_task.max(),
        #     "critic_can_beat_best_model_initial": revised_scores_for_task.max() > initial_scores_for_task[best_on_current_task_index],
        #     "critic_can_beat_best_model_after_critic": revised_scores_for_task.max() > revised_scores_for_task[best_on_current_task_index].max()
        # })
        # by_task_dict[task]["critic_can_beat_best_model_max"] = by_task_dict[task]["critic_can_beat_best_model_initial"] or by_task_dict[task]["critic_can_beat_best_model_after_critic"]



    return by_task_dict, all_models_sorted

    with open(path_to_scores, 'rt') as f:
        scores = json.load(f)
    labels_for_classifier = {}
    labels_for_multi_label = {}
    max_improvement_dict = {}
    for model in scores:
        labels_for_classifier[model] = {}
        labels_for_multi_label[model] = {}
        max_improvement_dict[model] = {}
        for task in scores[model]:
            current_scores = np.array(scores[model][task])
            max_improvement = np.max(current_scores)
            max_improvement_dict[model][task] = max_improvement
            labels_for_classifier[model][task] = int(max_improvement > IMPROVEMENT_THRESHOLD)
            multiclass_labels = np.zeros_like(current_scores)
            multiclass_labels[current_scores==max_improvement] = 1
            labels_for_multi_label[model][task] = multiclass_labels
    return labels_for_classifier, labels_for_multi_label, max_improvement_dict, scores

def prepare_dataset(by_task_dict, path_to_representation, path_to_tasks, only_model_data, model_index):

    with open(path_to_tasks, 'rt') as f:
        tasks_ordered = json.load(f)
    representations = np.load(path_to_representation)

    tasks_ordered_indices = [i for i, task in enumerate(tasks_ordered) if task in by_task_dict]
    tasks_ordered_with_label = [task for i, task in enumerate(tasks_ordered) if i in tasks_ordered_indices]
    representations_with_label = representations[tasks_ordered_indices]
    scores_ordered = [by_task_dict[task] for task in tasks_ordered_with_label]


    single_label_critic = [np.random.choice(np.where(scores_ordered[i]==max_improvement_ordered[i])[0]) for i in range(len(multilabel_labels))]
    classify_opposite = 1-np.array(classifier_labels)
    single_label_with_bad_class = np.array(single_label_critic)
    single_label_with_bad_class[classify_opposite==1] = -1
    dataset_dict = {
        "task": tasks_ordered_with_label,
        "representation": representations_with_label,
        "scores": scores_ordered,
        "classifier_labels": classifier_labels,
        "max_critic_improvement": max_improvement_ordered,
        "best_critic_multilabel": multilabel_labels,
        "best_critic_single_label": single_label_critic,
        "best_critic_with_minus_1": single_label_with_bad_class
    }
    ds = Dataset.from_dict(dataset_dict)
    return ds

def calculate_baselines(by_task_dict):

    by_task_df = pd.DataFrame(by_task_dict)
    best_init_score = by_task_df["init_score_for_best_gen_current_task"]
    critic_routing_best_on_train = by_task_df["best_revised_score_for_best_gen_on_train"]
    initial_score_best_on_train = by_task_df["init_score_for_best_gen_on_train"]
    best_on_train_only_gen_with_filtering = np.max([initial_score_best_on_train, critic_routing_best_on_train], axis=1)
    best_on_train_gen_and_critic_with_filtering = np.max([initial_score_best_on_train, gen_and_critic_are_best_on_train], axis=1)
    max_revised = by_task_df["max_revised_score"]

    for task in by_task_dict:
        task_dict_values = by_task_dict[task]

        # model routing
        best_init_score = task_dict_values["init_score_for_best_gen_current_task"]

        # best critic on train and best generator on train
        gen_and_critic_are_best_on_train = task_dict_values["critic_best_on_train_improving_best_on_train"]

        # always send to critic routing -- gen best on train
        critic_routing_best_on_train = task_dict_values["best_revised_score_for_best_gen_on_train"]

        # gen best on train -- critic or initial response
        initial_score_best_on_train = task_dict_values["init_score_for_best_gen_on_train"]
        best_on_train_only_gen_with_filtering = max([initial_score_best_on_train, critic_routing_best_on_train])

        # gen and critic are best on train -- critic or initial response
        best_on_train_gen_and_critic_with_filtering = max([initial_score_best_on_train, gen_and_critic_are_best_on_train])

        # upper bound send to critic
        max_revised = task_dict_values["max_revised_score"]

        # upper bound with filtering
        upper_bound_with_filtering = max([best_init_score, max_revised])



def main(path_to_config, split, combined: bool, data):
    with open(path_to_config, 'rt') as f:
        config = json.load(f)
    # path_to_scores = config["scores_path"].format(split=split, scoring_metric=config["scoring_metric"])
    by_task_dict, all_models_sorted = get_all_scores(data)
    all_ds = []
    by_task_df = pd.DataFrame(by_task_dict.values())

    # "model_routing_score": initial_scores_for_task.max(),
    # "best_on_train_label": best_on_train_label,
    # "best_on_train_critic_scores": best_on_train_critic_scores,
    # "best_on_train_upper_bound_score": best_on_train_critic_scores.max(),
    # "best_on_train_used_critic": best_on_train_label[-1] == 0,  # initial score is in the -1 index
    # "self_critic_for_best_on_train_score": best_on_train_critic_scores[best_on_train_index],
    # "initial_score_best_on_train": best_on_train_critic_scores[-1],



    for model in data["generator_model"].unique():
        only_model_df = data.loc[data["generator_model"]]
        # print model initial response scores
        # print model self critic scores no filtering
        # print model self critic scores with filtering
        # for model in all other models
        #   print model with critic scores no filtering
        #   print model with critic scores with filtering
        # print model critic routing

    print("UPPER BOUNDS AND BASELINES")
    print("model routing", by_task_df["model_routing_score"].mean().round(3))
    print('-----------------------')
    print("critic routing on best on train", by_task_df["best_on_train_upper_bound_score"].mean().round(3))
    print("self critic on best on train", by_task_df["self_critic_for_best_on_train_score"].mean().round(3))
    print("best critic on train, on best on train", by_task_df["best_critic_on_train_for_best_on_train_generation"].mean().round(3))
    print('-----------------------')
    print("generation+critic routing", by_task_df["gen_and_critic_routing_max_score"].mean().round(3))
    print('-----------------------')
    print("critic routing on best initial generation", by_task_df["critic_routing_on_model_routing"].mean().round(3))
    print("self critic on best initial generation", by_task_df["self_critic_on_model_routing"].mean().round(3))
    print("best critic on train, on best initial generation", by_task_df["best_critic_on_train_for_model_routing"].mean().round(3))

    print("\n\nStatistical Significance Tests")
    # In the paired samples t-test the null hypothesis is that the average of the differences between the paired observations in the two samples is zero.
    # If the calculated P-value is less than 0.05, the conclusion is that, statistically, the mean difference between the paired observations is significantly different from 0.
    print("\nPaired t-test -- Critic routing on best generation vs initial best generation (model routing):")
    print(stats.ttest_rel(by_task_df["model_routing_score"], by_task_df["critic_routing_on_model_routing"]))
    print("\nPaired t-test -- Critic routing on best generation vs self critic on best generation:")
    print(stats.ttest_rel(by_task_df["self_critic_on_model_routing"], by_task_df["critic_routing_on_model_routing"]))
    print("\nPaired t-test -- Critic routing on best generation vs generation+critic routing")
    print(stats.ttest_rel(by_task_df["critic_routing_on_model_routing"], by_task_df["gen_and_critic_routing_max_score"]))
    print("\nPaired t-test -- model routing vs critic routing on best on train:")
    print(stats.ttest_rel(by_task_df["model_routing_score"], by_task_df["best_on_train_upper_bound_score"]))

    # get all tasks representations
    embedding_dir = config["embedding_dir"].format(split=split, gen_model=all_models_sorted[0])
    path_to_representation = config["path_to_embeddings"].format(embedding_dir=embedding_dir,
                                                                 representation='tasks')
    task_representations = np.load(path_to_representation)

    # get tasks list
    path_to_tasks = config["path_to_tasks"].format(embedding_dir=embedding_dir)
    with open(path_to_tasks, 'rt') as f:
        tasks_ordered = json.load(f)

    tasks_ordered_indices = [i for i, task in enumerate(tasks_ordered) if task in by_task_dict]
    tasks_ordered_with_label = [task for i, task in enumerate(tasks_ordered) if i in tasks_ordered_indices]
    task_representations = task_representations[tasks_ordered_indices]

    # get all model outputs
    model_outputs_representations = []
    input_and_output_representations = []
    for model in all_models_sorted:
        embedding_dir = config["embedding_dir"].format(split=split, gen_model=model)
        path_to_representation = config["path_to_embeddings"].format(embedding_dir=embedding_dir,
                                                                     representation='outputs')
        repr_for_model = np.load(path_to_representation)[tasks_ordered_indices]
        model_outputs_representations.append(repr_for_model)
        path_to_representation = config["path_to_embeddings"].format(embedding_dir=embedding_dir,
                                                                     representation='input_and_output')
        repr_for_model = np.load(path_to_representation)[tasks_ordered_indices]
        input_and_output_representations.append(repr_for_model)
    # combined_algorithm_representation = np.hstack((task_representations, *np.array(model_outputs_representations)))
    # reduce_repr = PCA(n_components=1024)
    # print("RUNNING PCA FOR COMBINED REPRESENTATION")
    # reduce_repr = reduce_repr.fit_transform(combined_algorithm_representation)

    combined_labels_ordered = [by_task_dict[task]["combined_label_gen_and_critic"] for task in tasks_ordered_with_label]

    # two steps router
    two_steps_labels = [by_task_dict[task]["two_steps_label"] for task in tasks_ordered_with_label]
    # generator_labels = [get_models_from_combined_label(label, len(all_models_sorted))[0] for label in two_steps_labels]
    all_scores_ordered = np.array([by_task_dict[task]["combined_scores"] for task in tasks_ordered_with_label])

    data_dict = {
        "task": tasks_ordered_with_label,
        "task_representations": task_representations,
        "gen_and_critic_combined_labels": combined_labels_ordered,
        "two_steps_labels": two_steps_labels,
        "all_scores_ordered": all_scores_ordered
    }
    for i in range(len(all_models_sorted))  :
        data_dict[f"model_outputs_representations_{i}"] = model_outputs_representations[i]
        data_dict[f"input_and_output_representations_{i}"] = input_and_output_representations[i]
    ds = Dataset.from_dict(data_dict)
    ds.save_to_disk("/tmp/data_for_training_1201")
    return ds

def train(ds, all_models_sorted):

    split_ds = ds.train_test_split(test_size=0.3, seed=42)

    # train combination
    x_train = np.hstack((split_ds["train"]["task_representations"], *np.array([split_ds["train"][f"model_outputs_representations_{i}"] for i in range(5)])))
    y_train = split_ds["train"]["gen_and_critic_combined_labels"]
    x_test = np.hstack((split_ds["test"]["task_representations"], *np.array([split_ds["test"][f"model_outputs_representations_{i}"] for i in range(5)])))
    y_test = split_ds["test"]["gen_and_critic_combined_labels"]

    all_x = np.vstack((x_train, x_test))
    pca = PCA(n_components=1024)
    print("RUNNING PCA -- reducing dimension")
    all_x_reduced = pca.fit_transform(all_x)
    x_train = all_x_reduced[:len(x_train)]
    x_test = all_x_reduced[len(x_train):]
    print("PCA done")

    svc = SVC(kernel='poly')
    svc.fit(x_train, y_train)
    predictions = svc.predict(x_test)
    convert_predictions_to_indices = [get_models_from_combined_label(pred, num_models=5) for pred in predictions]
    revised_scores = [np.array(score)[convert_predictions_to_indices[i]] for i, score in enumerate(split_ds["test"]["all_scores_ordered"])]
    print("predicting critic and generator combined -- SVC poly", np.mean(revised_scores).round(3))

    knn = KNeighborsClassifier(n_neighbors=NUM_KS)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    convert_predictions_to_indices = [get_models_from_combined_label(pred, num_models=5) for pred in predictions]
    revised_scores = [np.array(score)[convert_predictions_to_indices[i]] for i, score in
                      enumerate(split_ds["test"]["all_scores_ordered"])]
    print(f"predicting critic and generator combined -- {NUM_KS}NN", np.mean(revised_scores).round(3))


    # train model routing
    x_train = split_ds["train"]["task_representations"]
    y_train_two_steps = split_ds["train"]["two_steps_labels"]
    y_train_model_routing = [get_models_from_combined_label(y, num_models=5)[0] for y in y_train_two_steps]
    x_test = split_ds["test"]["task_representations"]
    y_test_two_steps = split_ds["test"]["two_steps_labels"]
    y_test_model_routing = [get_models_from_combined_label(y, num_models=5)[0] for y in y_test_two_steps]

    svc = SVC(kernel='poly')
    svc.fit(x_train, y_train_model_routing)
    predictions_model_routing_svc = svc.predict(x_test)
    revised_scores = [np.array(score)[predictions_model_routing_svc[i], -1] for i, score in
                      enumerate(split_ds["test"]["all_scores_ordered"])]
    print("predicting model routing -- SVC poly", np.mean(revised_scores).round(3))

    knn = KNeighborsClassifier(n_neighbors=NUM_KS)
    knn.fit(x_train, y_train_model_routing)
    predictions_model_routing_knn = knn.predict(x_test)
    revised_scores = [np.array(score)[predictions_model_routing_knn[i], -1] for i, score in
                      enumerate(split_ds["test"]["all_scores_ordered"])]
    print(f"predicting model routing -- {NUM_KS}NN", np.mean(revised_scores).round(3))

    model_predictions = {}
    for i, model in enumerate(all_models_sorted):
        # train critic router
        x_train = split_ds["train"][f"input_and_output_representations_{i}"]
        y_train_critic_router = [np.array(score)[i].argmax() for score in split_ds["train"]["all_scores_ordered"]]
        x_test = split_ds["test"][f"input_and_output_representations_{i}"]
        y_test_critic_router = [np.array(score)[i].argmax() for score in split_ds["test"]["all_scores_ordered"]]

        svc = SVC(kernel='poly')
        svc.fit(x_train, y_train_critic_router)
        predictions_critic_routing_svc = svc.predict(x_test)
        revised_scores = [np.array(score)[i, predictions_critic_routing_svc[j]] for j, score in
                          enumerate(split_ds["test"]["all_scores_ordered"])]
        print(f"predicting critic routing for {model} -- SVC poly", np.mean(revised_scores).round(3))

        knn = KNeighborsClassifier(n_neighbors=NUM_KS)
        knn.fit(x_train, y_train_critic_router)
        predictions_critic_routing_knn = svc.predict(x_test)
        revised_scores = [np.array(score)[i, predictions_critic_routing_knn[j]] for j, score in
                          enumerate(split_ds["test"]["all_scores_ordered"])]
        print(f"predicting critic routing for {model} -- {NUM_KS}NN", np.mean(revised_scores).round(3))
        model_predictions[model] = {f"{NUM_KS}nn": predictions_critic_routing_knn, "svc_poly": predictions_critic_routing_svc}

    two_step_predictions_svc = []
    two_step_predictions_knn = []
    for i, task in enumerate(split_ds["test"]):
        model_routing = predictions_model_routing_svc[i]
        model_name = all_models_sorted[model_routing]
        critic_routing_for_model = model_predictions[model_name]["svc_poly"][i]
        revised_score_for_pred = np.array(task["all_scores_ordered"])[model_routing, critic_routing_for_model]
        two_step_predictions_svc.append(revised_score_for_pred)

        model_routing = predictions_model_routing_knn[i]
        model_name = all_models_sorted[model_routing]
        critic_routing_for_model = model_predictions[model_name][f"{NUM_KS}nn"][i]
        revised_score_for_pred = np.array(task["all_scores_ordered"])[model_routing, critic_routing_for_model]
        two_step_predictions_knn.append(revised_score_for_pred)

    print(f"two-step router {NUM_KS}nn", np.mean(two_step_predictions_knn).round(3))
    print(f"two-step router svc", np.mean(two_step_predictions_svc).round(3))

    return



    critic_labels = [score[i].argmax() for score in all_scores_ordered]
    x = ds["input_and_output_representation"][i]
    y = critic_labels
    # separate classifier for each model
    for i, model in enumerate(all_models_sorted):
        embedding_dir = config["embedding_dir"].format(split=split, gen_model=model)
        path_to_representation = config["path_to_embeddings"].format(embedding_dir=embedding_dir,
                                                                     representation=config["representation"])
        if not os.path.exists(path_to_representation):
            continue
        representation = np.load(path_to_representation)
        path_to_tasks = config["path_to_tasks"].format(embedding_dir=embedding_dir)
        if not os.path.exists(path_to_tasks):
            continue
        only_model_data = data.loc[data["generator_model"] == model].reset_index(False)

    # combined training
    # each task is represented as a concatenation of [task, resp1, resp2, ..., resp5]
    # labels: combined_label_gen_and_critic

    for i, model in enumerate(all_models_sorted):
        embedding_dir = config["embedding_dir"].format(split=split, gen_model=model)
        path_to_representation = config["path_to_embeddings"].format(embedding_dir=embedding_dir,
                                                                     representation=config["representation"])
        if not os.path.exists(path_to_representation):
            continue
        path_to_tasks = config["path_to_tasks"].format(embedding_dir=embedding_dir)
        if not os.path.exists(path_to_tasks):
            continue
        only_model_data = data.loc[data["generator_model"] == model].reset_index(False)
        ds = prepare_dataset(by_task_dict, path_to_representation, path_to_tasks, only_model_data, model_index=i)
        ds = ds.add_column('model', [model]*len(ds))
        all_ds.append(ds)
    if combined:
        combined_ds = datasets.concatenate_datasets(all_ds)
        all_ds = [combined_ds]

    can_be_improved = lambda s: s["max_critic_improvement"] > IMPROVEMENT_THRESHOLD
    all_models_sorted = sorted(list(max_improvement.keys()))
    for ds in all_ds:
        all_eval_metrics = {}
        for _ in range(NUM_ROUNDS):
            # split
            split_ds = ds.train_test_split(test_size=0.2)
            train = split_ds["train"]
            test = split_ds["test"]

            # filter by gold
            filtered_to_only_improved_train = train.filter(can_be_improved)
            filtered_to_only_improved_test = test.filter(can_be_improved)

            # baselines filtered
            filtered_baselines = calculate_baselines(filtered_to_only_improved_test, filtered_to_only_improved_train, all_models_sorted)
            for k in filtered_baselines:
                if k not in all_eval_metrics["filtered"]:
                    all_eval_metrics["filtered"][k] = []
                all_eval_metrics["filtered"][k].append(filtered_baselines[k])
            # baselines not filtered
            unfiltered_baselines = calculate_baselines(test, train, all_models_sorted)
            for k in unfiltered_baselines:
                if k not in all_eval_metrics["unfiltered"]:
                    all_eval_metrics["unfiltered"][k] = []
                all_eval_metrics["unfiltered"][k].append(unfiltered_baselines[k])

            # knn
            for k in range(NUM_KS, NUM_KS+1):
                name = f"{k}NN"
                knn = KNeighborsClassifier(n_neighbors=k)
                filtered_pred_scores = fit_and_predict(knn, filtered_to_only_improved_train, filtered_to_only_improved_test)
                if name not in all_eval_metrics["filtered"]:
                    all_eval_metrics["filtered"][name] = []
                all_eval_metrics["filtered"][name].append(filtered_pred_scores)
                knn = KNeighborsClassifier(n_neighbors=k)
                unfiltered_pred_scores = fit_and_predict(knn, train, test)
                if name not in all_eval_metrics["unfiltered"]:
                    all_eval_metrics["unfiltered"][name] = []
                all_eval_metrics["unfiltered"][name].append(unfiltered_pred_scores)

            # svc
            for kernel in ['poly']:#['poly', 'linear', 'sigmoid', 'rbf']:
                for c in [1.2]: #[0.5, 0.8, 1, 1.2, 1.5, 2]:
                    svc = SVC(kernel=kernel, C=c)
                    filtered_pred_scores = fit_and_predict(svc, filtered_to_only_improved_train, filtered_to_only_improved_test)
                    name = f"{kernel}-{c}"
                    if name not in all_eval_metrics["filtered"]:
                        all_eval_metrics["filtered"][name] = []
                    all_eval_metrics["filtered"][name].append(filtered_pred_scores)
                    svc = SVC(kernel=kernel, C=c)
                    unfiltered_pred_scores = fit_and_predict(svc, train, test)
                    if name not in all_eval_metrics["unfiltered"]:
                        all_eval_metrics["unfiltered"][name] = []
                    all_eval_metrics["unfiltered"][name].append(unfiltered_pred_scores)
        print(f"{config['representation']} representation")
        if combined:
            print("COMBINED DATASET")
        else:
            print(ds["model"][0].upper())
        for cat in ["filtered", "unfiltered"]:
            print('\n\n',"========",cat.upper(),"=======")
            for k in all_eval_metrics[cat]:
                print(k, np.mean(all_eval_metrics[cat][k]).round(2))
        # calculate mean
    # splitted_ds = combined_ds.train_test_split(test_size=0.2)
    # train = splitted_ds["train"]
    # test = splitted_ds["test"]
    # filtered_to_only_improved_train = train.filter(lambda sample: sample["max_critic_improvement"] > IMPROVEMENT_THRESHOLD)
    # filtered_to_only_improved_test = test.filter(lambda sample: sample["max_critic_improvement"] > IMPROVEMENT_THRESHOLD)
    #
    # all_eval_metrics = {}
    # print("BASELINES")
    # print("filtered")
    # print("all data")
    #
    # for k in range(1,11):
    #     print(f"\n\nKNN with K={k}")
    #     print("filtered", end=' ')
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     fit_and_predict(knn, filtered_to_only_improved_train, filtered_to_only_improved_test)
    #     print("all data", end=' ')
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     fit_and_predict(knn, train, test)
    #
    # for kernel in ['poly', 'linear', 'sigmoid']:
    #     print(f"\n\nSVC {kernel}")
    #     print("filtered", end=' ')
    #     svc = SVC(kernel=kernel)
    #     fit_and_predict(svc, filtered_to_only_improved_train, filtered_to_only_improved_test)
    #     print("all data", end=' ')
    #     svc = SVC(kernel=kernel)
    #     fit_and_predict(svc, train, test)

def fit_and_predict(trainer, train, test):
    trainer.fit(train["representation"], train["best_critic_with_minus_1"])
    preds = trainer.predict(test["representation"])
    scores = calc_scores(test, preds)
    return scores

def calculate_baselines(test, train, all_models_sorted):
    results = {}
    scores = calc_scores(test, test["best_critic_with_minus_1"])
    results["upper_bound"] = scores

    max_train = Counter(train["best_critic_single_label"]).most_common(1)[0][0]
    baseline_max_train = np.full_like(test["best_critic_single_label"], fill_value=max_train)
    scores = calc_scores(test, baseline_max_train)
    results["best_on_train"] = scores

    pred_self_critic = [all_models_sorted.index(sample['model']) for sample in test]
    scores = calc_scores(test, pred_self_critic)
    results["self_critic"] = scores

    return results


def baselines(ds, models_sorted):
    all_scores = np.array(ds["all_scores_ordered"])
    for generator_model in models_sorted:
        gen_model_index = models_sorted.index(generator_model)
        scores_for_gen_model = all_scores[:, gen_model_index, :]
        print("\n\nGENERATOR", generator_model)
        print("Init score", scores_for_gen_model[:, -1].mean().round(3))
        print("Self-critic (no filtering)", scores_for_gen_model[:, gen_model_index].mean().round(3))
        self_critic_with_filtering = np.max(scores_for_gen_model[:, [gen_model_index, -1]], axis=1)
        print("Self-critic with classifier", self_critic_with_filtering.mean().round(3))
        for critic_model in models_sorted:
            if critic_model == generator_model:
                continue
            critic_model_index = models_sorted.index(critic_model)
            print("Critic by", critic_model, "(no filtering)", scores_for_gen_model[:, critic_model_index].mean().round(3))
            static_critic_with_filtering = np.max(scores_for_gen_model[:, [critic_model_index, -1]], axis=1)
            print("Critic by", critic_model, "with classifier", static_critic_with_filtering.mean().round(3))
        critic_routing = np.max(scores_for_gen_model, axis=1)
        print("Critic routing", critic_routing.mean().round(3))
    best_gen_model = all_scores[:,:,-1].argmax(axis=1)
    only_best_gen_scores = get_only_best_gen_critic_scores(all_scores)
    print("\n\nModel Routing (initial response score)", all_scores[:,:,-1].max(axis=1).mean().round(3))
    self_critic_on_model_routing = np.take_along_axis(only_best_gen_scores, best_gen_model.reshape(-1,1), axis=1).squeeze()
    print("Self-critic on model routing (no filter)", self_critic_on_model_routing.mean().round(3))
    indices_with_classifier = np.vstack((best_gen_model, np.full_like(best_gen_model, fill_value=-1))).T
    self_critic_with_filtering_on_model_routing = np.take_along_axis(only_best_gen_scores, indices_with_classifier, axis=1).max(axis=1)
    print("Self-critic on model routing with classifier", self_critic_with_filtering_on_model_routing.mean().round(3))
    for critic_model in models_sorted:
        critic_model_index = models_sorted.index(critic_model)
        print("Critic by", critic_model, "(no filtering) on model routing", only_best_gen_scores[:, critic_model_index].mean().round(3))
        static_critic_with_filtering = np.max(only_best_gen_scores[:, [critic_model_index, -1]], axis=1)
        print("Critic by", critic_model, "with classifier on model routing", static_critic_with_filtering.mean().round(3))
    critic_routing = np.max(only_best_gen_scores, axis=1)
    print("Critic routing on model routing", critic_routing.mean().round(3))


def get_only_best_gen_critic_scores(all_scores):
    best_gen_model = all_scores[:,:,-1].argmax(axis=1)
    only_best_gen_scores = []
    for sample_index, model_index in enumerate(best_gen_model):
        only_best_gen_scores.append(all_scores[sample_index, model_index])
    only_best_gen_scores = np.array(only_best_gen_scores)
    return only_best_gen_scores

def get_y_for_split_scores(split_data):
    all_scores = np.array(split_data["all_scores_ordered"])
    only_best_gen_scores = get_only_best_gen_critic_scores(all_scores)
    y_exp = np.exp(only_best_gen_scores)
    y = y_exp / np.vstack([np.sum(y_exp, axis=1)] * 6).T
    return y

def get_y_for_regression(split_data):
    all_scores = np.array(split_data["all_scores_ordered"])
    only_best_gen_scores = get_only_best_gen_critic_scores(all_scores)
    y = only_best_gen_scores[:, :5] - only_best_gen_scores[:, 5].reshape(-1,1)
    return y

def get_x_for_split_scores(split_data, representation):
    all_scores = np.array(split_data["all_scores_ordered"])
    best_gen_model = all_scores[:, :, -1].argmax(axis=1)
    if representation in split_data.column_names:
        x = split_data[representation]
    else:
        all_reps  = [split_data[f"{representation}_{model}"] for model in range(5)]
        all_reps = np.array(all_reps)
        x = [all_reps[model, sample_index] for sample_index, model in enumerate(best_gen_model)]
    x = np.array(x)
    return x

def eval_pred_kl(estimator, X_test, y_test):
    preds = estimator.predict(X_test)
    preds = np.exp(preds)
    scores = []
    for t, p in zip(preds, y_test):
        t = t / sum(t)
        kl_div = entropy(t, p)
        if kl_div == np.inf:
            print("kl div inf", t, p)
            continue
        scores.append(kl_div)
    scores = np.array(scores)
    scores = scores[scores!=np.inf]
    return -np.sum(scores)

def eval_pred_f1(estimator, X_test, y_test):
    preds = estimator.predict(X_test)
    preds = np.argmax(preds, axis=1)
    sample_weight = np.max(y_test, axis=1)
    y_test = np.argmax(y_test, axis=1)
    f1 = f1_score(y_test, preds, average='macro', sample_weight=sample_weight)
    return f1

def eval_pred_svc(estimator, X_test, y_test):
    preds = estimator.predict(X_test)
    weights = np.ones(len(y_test))
    weights[y_test == 5] *= 53.9
    weights[y_test == 4] *= 47.6
    weights[y_test == 3] *= 50.1
    weights[y_test == 2] *= 46.9
    weights[y_test == 1] *= 48.2
    weights[y_test == 0] *= 48.5
    f1 = f1_score(y_test, preds, average='macro', sample_weight=weights)
    return f1

def hyper_parameter_search(ds_for_hps):
    y = get_y_for_split_scores(ds_for_hps)
    sample_weight = y.max(axis=1)

    for representation_key in ["task_representations", "input_and_output_representations", "model_outputs_representations"]:
        print("\n\n---------------------", representation_key, "---------------------", sep='\n')
        x = get_x_for_split_scores(ds_for_hps, representation_key)

        print("\nHP search for SVR...")

        svr = SVR()
        multi_output_regressor = MultiOutputRegressor(svr)
        param_grid = {
            'estimator__kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
            'estimator__C': [1e-3, 1e-2, 1e-1, 1, 10],
            'estimator__gamma': [0.1, 0.5, 1, 2, 5]
        }
        grid_search = GridSearchCV(multi_output_regressor, param_grid, cv=10, scoring=eval_pred_f1, verbose=1, n_jobs=-1)
        grid_search.fit(x, y, sample_weight=sample_weight)
        print("Best parameters:", grid_search.best_params_)
        print("Best score:", grid_search.best_score_)

        print("\nHP search for SVC...")
        # svc = SVC(class_weight={5: 53.9, 4: 47.6, 3: 50.1, 2: 46.9, 1: 48.2, 0: 48.5})
        svc = SVC()
        grid_search = GridSearchCV(svc, {k.replace('estimator__', ''): param_grid[k] for k in param_grid}, cv=5, scoring=eval_pred_svc, verbose=1, n_jobs=-1)
        grid_search.fit(x, y.argmax(axis=1), sample_weight=sample_weight)
        print("Best parameters:", grid_search.best_params_)
        print("Best score:", grid_search.best_score_)


        print("\nHP search for KNN regressor...")

        knn = KNeighborsRegressor()
        multi_output_regressor = MultiOutputRegressor(knn)
        param_grid = {
            'estimator__n_neighbors': [3, 5, 7],
            'estimator__weights': ['uniform', 'distance'],
            'estimator__metric': ['euclidean', 'manhattan'],
            'estimator__leaf_size': [10, 30],
            'estimator__algorithm': ['auto', 'ball_tree'],
            'estimator__p': [1, 2]
        }
        grid_search = GridSearchCV(multi_output_regressor, param_grid, cv=5, scoring=eval_pred_f1, verbose=1, n_jobs=-1)
        grid_search.fit(x, y)
        print("Best parameters:", grid_search.best_params_)
        print("Best score:", grid_search.best_score_)

        print("\nHP search for KNN classifier...")

        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, {k.replace('estimator__', ''): param_grid[k] for k in param_grid}, cv=5, scoring=eval_pred_f1, verbose=1, n_jobs=-1)
        grid_search.fit(x, y, sample_weight=sample_weight)
        print("Best parameters:", grid_search.best_params_)
        print("Best score:", grid_search.best_score_)


        # rf = RandomForestRegressor()
        # multi_output_regressor = MultiOutputRegressor(rf)
        # param_grid = {
        #     'estimator__n_estimators': [10, 50, 100],
        #     'estimator__max_depth': [5, 10, 15],
        #     'estimator__min_samples_split': [2, 5, 10],
        #     'estimator__min_samples_leaf': [1, 2, 4]
        # }
        # grid_search = GridSearchCV(multi_output_regressor, param_grid, cv=5, scoring=eval_pred_f1, verbose=True, n_jobs=-1)
        # grid_search.fit(x, y)
        # print("Best parameters:", grid_search.best_params_)
        # print("Best score:", grid_search.best_score_)

def train_critic_routing(ds):
    split_ds = ds.train_test_split(test_size=0.3, seed=42)
    only_best_gen_scores = get_only_best_gen_critic_scores(np.array(split_ds["test"]["all_scores_ordered"]))

    y_train = get_y_for_split_scores(split_ds["train"])
    y_test = get_y_for_split_scores(split_ds["test"])

    print("\n\ntrain labels distribution", Counter(y_train.argmax(axis=1)))
    print("test labels distribution",Counter(y_test.argmax(axis=1)))

    print("\nBASELINES")
    for cls in np.arange(6):
        predictions = np.full(len(y_test), fill_value=cls)
        revised_scores = [np.array(score)[predictions[i]] for i, score in
                          enumerate(only_best_gen_scores)]
        print(f"Predicting always {cls}", np.mean(revised_scores).round(3))

    optimal_c = {"task_representations": 10, "input_and_output_representations": 10, "model_outputs_representations": 10}
    optimal_gamma = {"task_representations": 5, "input_and_output_representations": 5, "model_outputs_representations": 5}
    optimal_kernel = {"task_representations": 'sigmoid', "input_and_output_representations": 'sigmoid', "model_outputs_representations": 'sigmoid'}
    optimal_algorithm = {"task_representations": 'auto', "input_and_output_representations": 'auto', "model_outputs_representations": 'auto'}
    optimal_leaf_size = {"task_representations": 10, "input_and_output_representations": 10, "model_outputs_representations": 10}
    optimal_metric = {"task_representations": 'manhattan', "input_and_output_representations": 'euclidean', "model_outputs_representations": 'euclidean'}
    optimal_n_neighbors = {"task_representations": 3, "input_and_output_representations": 3, "model_outputs_representations": 3}
    optimal_p = {"task_representations": 1, "input_and_output_representations": 1, "model_outputs_representations": 1 }
    optimal_weights = {"task_representations": 'distance', "input_and_output_representations": 'distance', "model_outputs_representations": 'distance'}

    sample_weight = y_train.max(axis=1)

    for representation_key in ["task_representations", "input_and_output_representations", "model_outputs_representations"]:
        print("\n\n---------------------", representation_key, "---------------------", sep='\n')
        x_train = get_x_for_split_scores(split_ds["train"], representation_key)
        x_test = get_x_for_split_scores(split_ds["test"], representation_key)
        class_weight = {5: 53.9, 4: 47.6, 3: 50.1, 2: 46.9, 1: 48.2, 0: 48.5}
        print('SVC weighted')
        cost_matrix = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                cost_matrix[i, j] = class_weight[i] / class_weight[j]

        svc = SVC(C=10, gamma=2, kernel='poly')
        svc.fit(x_train, y_train.argmax(axis=1), sample_weight=sample_weight)
        predictions = svc.predict(x_test)
        revised_scores = [np.array(score)[predictions[i]] for i, score in
                          enumerate(only_best_gen_scores)]
        print(
            f"SVC with rbf kernel, c=10, gamma=0.1:", np.mean(revised_scores).round(3))

        print("\nTrain optimal SVR params...")

        svr = SVR(C=optimal_c[representation_key], kernel=optimal_kernel[representation_key], gamma=optimal_gamma[representation_key])
        multi_output_regressor = MultiOutputRegressor(svr)
        multi_output_regressor.fit(x_train, y_train, sample_weight=sample_weight)
        predictions = multi_output_regressor.predict(x_test).argmax(axis=1)
        revised_scores = [np.array(score)[predictions[i]] for i, score in
                          enumerate(only_best_gen_scores)]
        print(f"SVR with {optimal_kernel[representation_key]} kernel, c={optimal_c[representation_key]}, gamma={optimal_gamma[representation_key]}:", np.mean(revised_scores).round(3))

        print("\nTrain optimal KNN params...")

        knn = KNeighborsRegressor(n_neighbors=optimal_n_neighbors[representation_key], algorithm=optimal_algorithm[representation_key],
                                  leaf_size=optimal_leaf_size[representation_key],
                                  p=optimal_p[representation_key], metric=optimal_metric[representation_key],
                                  weights=optimal_weights[representation_key])
        multi_output_regressor = MultiOutputRegressor(knn)
        multi_output_regressor.fit(x_train, y_train, sample_weight=sample_weight)
        predictions = multi_output_regressor.predict(x_test).argmax(axis=1)
        revised_scores = [np.array(score)[predictions[i]] for i, score in
                          enumerate(only_best_gen_scores)]
        print(f"KNN regressor with k={optimal_n_neighbors}", np.mean(revised_scores).round(3))

        knn = KNeighborsClassifier(n_neighbors=optimal_n_neighbors[representation_key],
                                  algorithm=optimal_algorithm[representation_key],
                                  leaf_size=optimal_leaf_size[representation_key],
                                  p=optimal_p[representation_key], metric=optimal_metric[representation_key],
                                  weights=optimal_weights[representation_key])
        knn.fit(x_train, y_train.argmax(axis=1), sample_weight=sample_weight)
        predictions = knn.predict(x_test).argmax(axis=1)
        revised_scores = [np.array(score)[predictions[i]] for i, score in
                          enumerate(only_best_gen_scores)]
        print(f"KNN classifier with k={optimal_n_neighbors}", np.mean(revised_scores).round(3))

        # print("Predicting critic routing on top of model routing results")
        #
        # print("  KNN")

        # for k in range(1, NUM_KS+1):
        #
        #     knn = KNeighborsClassifier(n_neighbors=k)
        #     knn.fit(x_train, y_train)
        #     predictions = knn.predict(x_test)
        #     revised_scores = [np.array(score)[predictions[i]] for i, score in
        #                       enumerate(only_best_gen_scores)]
        #     print(f"    k={k}", np.mean(revised_scores).round(3))

        # print('  SVC')
        # for kernel in ['poly', 'linear', 'sigmoid', 'rbf']:
        #     print(f'    {kernel}')
        #     for c in [0.1, 0.5, 0.8]:
        #         svc = SVC(kernel=kernel, C=c)
        #         svc.fit(x_train, y_train)
        #         predictions = svc.predict(x_test)
        #         revised_scores = [np.array(score)[predictions[i]] for i, score in
        #                           enumerate(only_best_gen_scores)]
        #         print(f"      c={c}", np.mean(revised_scores).round(3))
        #


        # for kernel in ['poly', 'linear', 'sigmoid', 'rbf']:
        #     print(f'    {kernel}')
        #     for c in [0.1, 0.5, 0.8]:
        #         svc = SVC(kernel=kernel, C=c)
        #         svc.fit(x_train, y_train, sample_weight=weights)
        #         predictions = svc.predict(x_test)
        #         revised_scores = [np.array(score)[predictions[i]] for i, score in
        #                           enumerate(only_best_gen_scores)]
        #         print(f"      c={c}", np.mean(revised_scores).round(3))


def train_regressors_for_each_class(ds):
    split_ds = ds.train_test_split(test_size=0.3, seed=42)
    only_best_gen_scores = get_only_best_gen_critic_scores(np.array(split_ds["test"]["all_scores_ordered"]))
    improvement = only_best_gen_scores[:, :5] - only_best_gen_scores[:, 5].reshape(-1,1)

    y_train = get_y_for_regression(split_ds["train"])
    y_test = get_y_for_regression(split_ds["test"])

    print("\nBASELINES")
    for cls in np.arange(6):
        predictions = np.full(len(y_test), fill_value=cls)
        revised_scores = [np.array(score)[predictions[i]] for i, score in
                          enumerate(only_best_gen_scores)]
        print(f"Predicting always {cls}", np.mean(revised_scores).round(3))

    for representation_key in ["input_and_output_representations", "model_outputs_representations"]:
        print("\n\n---------------------", representation_key, "---------------------", sep='\n')
        x_train = get_x_for_split_scores(split_ds["train"], representation_key)
        x_test = get_x_for_split_scores(split_ds["test"], representation_key)

        lr = LinearRegression()
        lr.fit(x_train, y_train)
        predictions = lr.predict(x_test)
        df = pd.DataFrame(predictions).T

        all_predictions = []
        all_train_pred = []
        for i in tqdm(range(5)):
            svr = SVR()
            svr.fit(x_train, y_train[:, i])
            predictions = svr.predict(x_test)
            pred_train = svr.predict(x_train)
            all_train_pred.append(pred_train)
            all_predictions.append(predictions)
        all_predictions.append(np.zeros_like(all_predictions[-1]))
        df = pd.DataFrame(all_train_pred).T
        df_label = pd.DataFrame(y_train)
        classification_predictions = np.argmax(all_predictions, axis=0)
        revised_scores = [np.array(score)[classification_predictions[i]] for i, score in
                          enumerate(only_best_gen_scores)]
        print("SVR with default params", np.mean(revised_scores).round(3))


        param_grid = {
            'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
            'C': [1e-3, 1e-2, 1e-1, 1, 10],
            'gamma': [0.1, 0.5, 1, 2, 5]
        }

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = [pool.apply_async(fit_predict, (k,c,g,i,x_train,y_train,x_test)) for k, c, g, i in product(param_grid['kernel'], param_grid['C'], param_grid['gamma'], range(5))]
            results = [result.get() for result in results]
        print(results)
        all_results = results[0]
        for i in range(1, len(results)):
            all_results.update(results[i])

        dict_by_indices = {idx: {} for idx in range(5)}
        for k in all_results:
            dict_by_indices[k[-1]][k[:-1]] = all_results[k]

        max_score = 0
        max_pred = None
        max_args = None
        class_5_threshold = 0
        for arg_keys in tqdm(dict_by_indices[0]):
            predictions = [dict_by_indices[i][arg_keys] for i in range(5)]
            predictions.append(np.full_like(predictions[-1], fill_value=0))
            classification_predictions = np.argmax(predictions, axis=0)
            revised_scores = [np.array(score)[classification_predictions[i]] for i, score in
                              enumerate(only_best_gen_scores)]
            score = np.mean(revised_scores)
            if score > max_score:
                max_score = score
                max_args = arg_keys
                max_pred = classification_predictions
                class_5_threshold = 0
            predictions[-1] = np.full_like(predictions[-1], fill_value=-0.05)
            classification_predictions = np.argmax(predictions, axis=0)
            revised_scores = [np.array(score)[classification_predictions[i]] for i, score in
                              enumerate(only_best_gen_scores)]
            score = np.mean(revised_scores)
            if score > max_score:
                max_score = score
                max_args = arg_keys
                max_pred = classification_predictions
                class_5_threshold = -0.05
        print(max_score, max_args, max_pred, class_5_threshold)


        all_predictions = []
        for i in tqdm(range(5)):
            svr = SVR()
            svr.fit(x_train, y_train[:,i])
            predictions = svr.predict(x_test)
            all_predictions.append(predictions)
        all_predictions.append(np.zeros_like(all_predictions[-1]))
        classification_predictions = np.argmax(all_predictions, axis=0)
        revised_scores = [np.array(score)[classification_predictions[i]] for i, score in
                          enumerate(only_best_gen_scores)]
        print("SVR with default params", np.mean(revised_scores).round(3))


import multiprocessing

def fit_predict(kernel, c, gamma, i, x_train, y_train, x_test):
    svr = SVR(kernel=kernel, C=c, gamma=gamma)
    svr.fit(x_train, y_train[:, i])
    predictions = svr.predict(x_test)
    return {(kernel, c, gamma, i): predictions}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--path_to_data")
    parser.add_argument("--split")
    parser.add_argument("--combined", default=False, action='store_true')
    args = parser.parse_args()
    if os.path.exists("/tmp/data_for_training_1201"):
        dataset = Dataset.load_from_disk("/tmp/data_for_training_1201")
        data_df = pd.read_csv(args.path_to_data)
        baselines(dataset, sorted(data_df["generator_model"].unique()))
        split_ds = dataset.train_test_split(train_size=2000)
        # hyper_parameter_search(split_ds["train"])
        # train_critic_routing(split_ds["test"])
        train_regressors_for_each_class(split_ds["test"])
        exit(0)
        train(dataset, sorted(data_df["generator_model"].unique()))

    if os.path.exists(args.path_to_data):
        data_df = pd.read_csv(args.path_to_data)
    else:
        improvement_pairs = parse_improvement_pairs(args.improvement_pairs, args.split)
        data_df = prepare_df(improvement_pairs, args.path_to_data)
    main(args.config, args.split, args.combined, data_df)
    if os.path.exists("/tmp/data_for_training_1201"):
        dataset = Dataset.load_from_disk("/tmp/data_for_training_1201")
        train(dataset)


