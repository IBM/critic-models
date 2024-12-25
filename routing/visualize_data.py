import json
import os
from argparse import ArgumentParser
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import re


COLORS = ["gray", "green", "blue", "orange", "pink", "red"]
# np.random.shuffle(COLORS)
ALPHA = 0.5

def visualize(config, split, representation_by, out_dir):
    df = pd.read_csv(config["path_to_tasks_df"])
    sorted_models = sorted(config["critic_models"])
    all_representations = {}
    all_tasks = {}

    for model in sorted_models:
        path_to_representations = config["representations_path"].format(embedding_dir=config["embedding_dir"],
                                                                        split=split,
                                                                        generator_model=model,
                                                                        representation=representation_by)
        path_to_tasks = os.path.join(os.path.dirname(path_to_representations), "sorted_tasks.json")
        all_representations[model] = np.load(path_to_representations)
        with open(path_to_tasks, 'rt') as t:
            all_tasks[model] = json.load(t)

    representations_for_best_gen = []
    for i, row in df.iterrows():
        task = row["task"]
        generator_model_routing_name = row["model_routing_name"]
        task_index_in_reps = all_tasks[generator_model_routing_name].index(task)
        representations_for_best_gen.append(all_representations[generator_model_routing_name][task_index_in_reps])
    df["representations"] = representations_for_best_gen

    print("RUNNING PCA...")
    representations_projection = PCA(n_components=2).fit_transform(representations_for_best_gen)

    legend_elements = [Line2D([0], [0], marker='.', markerfacecolor=COLORS[ind], label=sorted_models[ind],
                              color='w', markersize=16) for ind in range(len(sorted_models))]
    legend_elements.append(Line2D([0], [0], marker='.', markerfacecolor=COLORS[len(sorted_models)], label="No critic",
                              color='w', markersize=16))

    figure_name = "representation-by-generator-model"
    labels = df["model_routing_index"]
    colors_ordered = [COLORS[m] for m in labels.values]
    plot_and_save(representations_projection, figure_name, legend_elements[:-1], out_dir, colors_ordered)

    only_model_routing_scores = [eval(re.sub(r'\s+', ',', df["only_model_routing_scores"][i])) for i in range(len(df))]
    only_model_routing_scores = np.array(only_model_routing_scores)

    figure_name = "representation-by-best-critic-model"
    best_revised_scores = np.max(only_model_routing_scores, axis=1)
    best_critics_scores = only_model_routing_scores==best_revised_scores.reshape(-1,1)
    all_best_critics = []
    one_best_critic = []
    for c in best_critics_scores:
        best_models = np.where(c)[0]
        all_best_critics.append(best_models)
        one_best_critic.append(np.random.choice(best_models))
    colors_ordered = [COLORS[l] for l in one_best_critic]
    plot_and_save(representations_projection, figure_name, legend_elements, out_dir, colors_ordered)

    # todo plot different graph to different generator models



    figure_name = "init-scores"
    init_scores = only_model_routing_scores[:, -1]
    plot_and_save_with_colorbar(representations_projection, figure_name, out_dir, init_scores,
                                cmap='magma')


    figure_name = "revision-max-scores"
    revision_max_scores = only_model_routing_scores.max(axis=1)
    plot_and_save_with_colorbar(representations_projection, figure_name, out_dir, revision_max_scores,
                                cmap='magma')

    figure_name = "improvement-diff"
    diff = revision_max_scores-init_scores
    plot_and_save_with_colorbar(representations_projection, figure_name, out_dir, diff, cmap='cividis')

    figure_name = "improvement-diff-only-improved"
    plot_and_save_with_colorbar(representations_projection[diff>0], figure_name, out_dir, diff[diff>0], cmap='cividis')


    figure_name = "relative-improvement"
    init_scores_with_invalid = init_scores
    init_scores_with_invalid[init_scores<=0.1] = 0.1
    revision_scores_with_invalid = revision_max_scores
    revision_scores_with_invalid[np.bitwise_and(init_scores <= 0.1 ,revision_max_scores <= 0.1)] = 0.1
    relative = np.divide(revision_max_scores, init_scores_with_invalid)
    relative[relative>3] = 3
    plot_and_save_with_colorbar(representations_projection, figure_name, out_dir, relative, cmap='cividis')

    figure_name = "relative-improvement-only-improved"
    plot_and_save_with_colorbar(representations_projection[relative>1], figure_name, out_dir, relative[relative>1], cmap='cividis')


    figure_name = "self-critic"
    generator_models = df["model_routing_index"]
    self_critic_colors = ["#fc8d59", "#99d594", "#ffffbf"]
    self_critic_labels = []
    improved_indices = []
    for i, best_critics in enumerate(all_best_critics):
        if generator_models.iloc[i] in best_critics:
            self_critic_labels.append(1)
            improved_indices.append(i)
        elif np.all(len(sorted_models) == best_critics):
            self_critic_labels.append(2)
        else:
            self_critic_labels.append(0)
            improved_indices.append(i)
    colors_ordered = [self_critic_colors[label] for label in self_critic_labels]
    legend_elements_classification = [
        Line2D([0], [0], marker='.', markerfacecolor='#fc8d59', label='other-model-critic', color='w', markersize=16),
        Line2D([0], [0], marker='.', markerfacecolor='#99d594', label='self-critic', color='w', markersize=16),
        Line2D([0], [0], marker='.', markerfacecolor='#ffffbf', label='no-critic', color='w', markersize=16),
    ]
    plot_and_save(representations_projection, figure_name, legend_elements_classification, out_dir, colors_ordered)

    figure_name = "self-critic-only-improved"
    plot_and_save(representations_projection[improved_indices], figure_name,
                  legend_elements_classification[:-1], out_dir, np.array(colors_ordered)[improved_indices].tolist())


    figure_name = "distribution-of-init-vs-revised-by-best-generator"
    by_task_scores_groups = df.groupby("model_routing_name").groups
    plt.figure(figure_name)
    bins = np.linspace(0, 1.1, 21)

    for model in by_task_scores_groups:
        plt.figure(f"{figure_name}-{model}")
        init = only_model_routing_scores[by_task_scores_groups[model]][:,-1]
        counts_init, _ = np.histogram(init, bins=bins)
        revised = only_model_routing_scores[by_task_scores_groups[model]].max(axis=1)
        counts_revised, _ = np.histogram(revised, bins=bins)
        plt.plot(bins[:-1], counts_init, drawstyle='steps-post', color=self_critic_colors[0], label="initial")
        plt.plot(bins[:-1], counts_revised, drawstyle='steps-post', color=self_critic_colors[1], label="revised")
        plt.title(f"{figure_name}: {model.upper()}")
        plt.xlabel("Score")
        plt.ylabel('Count')
        plt.legend(loc='upper center')
        path = os.path.join(out_dir, f"{figure_name}-{model}.png")
        plt.savefig(path)

    figure_name = "distribution-of-init-vs-revised-by-best-critic"
    best_critic_indices = {j: [] for j in range(len(sorted_models))}
    for sample_index, best_critics in enumerate(all_best_critics):
        if len(sorted_models) in best_critics:  # init response no improvement
            continue
        for j in best_critics:
            best_critic_indices[j].append(sample_index)

    for model_index in best_critic_indices:
        indices = best_critic_indices[model_index]
        model_name = sorted_models[model_index]
        plt.figure(f"{figure_name}-{model_name}")
        init = only_model_routing_scores[indices][:, -1]
        counts_init, _ = np.histogram(init, bins=bins)
        revised = only_model_routing_scores[indices].max(axis=1)
        counts_revised, _ = np.histogram(revised, bins=bins)
        plt.plot(bins[:-1], counts_init, drawstyle='steps-post', color=self_critic_colors[0], label="initial")
        plt.plot(bins[:-1], counts_revised, drawstyle='steps-post', color=self_critic_colors[1], label="revised")
        plt.title(f"{figure_name}: {model_name.upper()}")
        plt.xlabel("Score")
        plt.ylabel('Count')
        plt.legend(loc='upper center')
        path = os.path.join(out_dir, f"{figure_name}-{model_name}.png")
        plt.savefig(path)



def plot_and_save(representations_projection, figure_name, legend_elements, out_dir, colors_ordered):
    plt.figure(figure_name)
    plt.scatter(representations_projection[:, 0], representations_projection[:, 1], c=colors_ordered, alpha=ALPHA)
    plt.legend(handles=legend_elements, loc='upper center')
    plt.title(figure_name)
    path = os.path.join(out_dir, f"{figure_name}.png")
    plt.savefig(path)

def plot_and_save_with_colorbar(representations_projection, figure_name, out_dir, colors_ordered, cmap):
    plt.figure(figure_name)
    plt.scatter(representations_projection[:, 0], representations_projection[:, 1], c=colors_ordered, alpha=ALPHA, cmap=cmap)
    plt.title(figure_name)
    plt.colorbar()
    path = os.path.join(out_dir, f"{figure_name}.png")
    plt.savefig(path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--out_dir")
    parser.add_argument("--representation")
    parser.add_argument("--split")
    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        config_json = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    visualize(config_json, args.split, args.representation, args.out_dir)