import os.path
import numpy as np
import json
from datasets import load_dataset
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
PATH_TO_GENERATIONS = "/Users/gililior/research/datasets/arena_data_v2/initial_generations_combined/"
BIG_MODELS = "init-generations-via-rits-{model}.constested-lmsys-chat-1m.json"
SMALL_MODELS = "{model}-it-test-init-gen.json"
from utils.utils import generate_color_map

CSV_PATH = "/Users/gililior/research/datasets/arena_data_v2/categories_single.csv"
DOMAINS_PATH = "/Users/gililior/research/datasets/arena_data_v2/domains.csv"
EMBEDDINGS_DIR = "/Users/gililior/research/datasets/arena_data_final/embeddings/{dump}/NV-Embed-v2-gemma-2-9b/"
CONSTRAINTS_LIST = os.path.join(EMBEDDINGS_DIR, "sorted_atomics.json")
CONSTRAINTS_EMBEDDINGS = os.path.join(EMBEDDINGS_DIR, "atomics.npy")
GENERATIONS_DIR = "/Users/gililior/research/datasets/arena_data_v2/generations/{dump}/"

DUMP = "test"

SCORES_DIR = "/Users/gililior/research/datasets/arena_data_v2/llm_aaj/{step}"
SCORES_PREFIX = "decomposition-evaluation-llama3.1-70b."
PATH = os.path.join(SCORES_DIR, SCORES_PREFIX)



def calculate(paths):

    color_map_by_model = generate_color_map()
    all_jsons = {}
    ds = load_dataset("gililior/wild-if-eval", split="test")
    # filter tasks with more than 8 constraints

    print("number of tasks", len(ds))
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    token_count = [len(tokenizer.encode(text)) for text in ds["task"]]
    all_constraints = ds["decomposition"]
    all_lengths = [len(c) for c in all_constraints]
    # correlation between token count and length of constraints
    print("correlation between token count and constraints per task", np.corrcoef(token_count, all_lengths)[0, 1])
    print("mean length of constraints", np.mean(all_lengths))
    # for each task, divide the length in tokens to the number of constraints

    all_constraints = [item for sublist in all_constraints for item in sublist]
    count_freq = Counter(all_constraints)
    count_counts = Counter(count_freq.values())
    keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"]
    values = []
    for i in range(1, 10):
        values.append(count_counts[i])
    # values.append(sum([count_counts[i] for i in range(8, 29)]))
    # values.append(sum([count_counts[i] for i in range(20, 41)]))
    values.append(sum([count_counts[i] for i in range(10, max(count_counts.keys()) + 1)]))
    # values[-1] = np.log(values[-1])
    plt.bar(np.arange(1, 11), values, color='#1f77b4', edgecolor='black')
    plt.yscale('log')
    plt.xticks(range(1, 11), keys)
    plt.xlabel("Number of occurrences of a constraint in the dataset", fontsize=12)
    plt.ylabel("Number of unique constraints (log scale)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("_output", "frequency_of_constraints.png"))

    plt.figure("histogram of num constraints")
    plt.bar(range(1, 9), [all_lengths.count(i) for i in range(1, 9)], color='#ff7f0e', edgecolor='black')
    plt.xticks(range(1, 9), [f"{i}" for i in range(1, 9)])
    plt.ylabel("Tasks count", fontsize=12)
    plt.xlabel("Constraints per task", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("_output", "histogram_of_num_constraints.png"))

    # Load data
    domain_df = pd.read_csv(DOMAINS_PATH)
    domain_count = Counter(domain_df["domain"])
    domain_count.pop("Artificial Intelligence")
    ordered_categories = sorted(domain_count.keys(), key=lambda x: domain_count[x], reverse=True)
    ordered_counts = [domain_count[cat] for cat in ordered_categories]
    ordered_categories = [cat if not pd.isna(cat) else "Other" for cat in ordered_categories]
    colors = plt.cm.Set3.colors
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(
        ordered_counts, labels=ordered_categories, autopct='%1.1f%%',
        colors=colors, startangle=0,
        wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 14},
        labeldistance=1.05,  # Moves labels closer to the pie
        pctdistance=0.85  # Moves percentage labels slightly inward
    )
    for i, text in enumerate(texts):
        text.set_bbox(dict(facecolor='none', edgecolor='none', alpha=0.75))  # Background for readability
    plt.margins(0)
    plt.tight_layout()
    output_path = os.path.join("_output", "domain_piechart.png")
    plt.savefig(output_path, dpi=300)  # Save with high resolution

    all_unique_constraints = set(all_constraints)
    print("number of unique constraints", len(all_unique_constraints))
    print("num of tasks", len(ds["task"]))

    for k in paths:
        with open(paths[k], 'r') as f:
            preds = json.load(f)
        all_jsons[k] = {task: preds[task] for task in ds["task"]}

    all_scores, all_constraints_scores, constraint_to_category = get_all_scores(all_jsons, token_count, ds, domain_df)

    # all_constraints_scores_for_validation = []
    # for model in all_constraints_scores:
    #     df = all_constraints_scores[model]
    #     df["model"] = model
    #     all_constraints_scores_for_validation.append(df)
    # all_constraints_scores_flatten = pd.concat(all_constraints_scores_for_validation)
    # binary_scores = [1 if s >= 0.5 else 0 for s in all_constraints_scores_flatten["score"]]
    # all_constraints_scores_flatten["binary_score"] = binary_scores
    # # choose random 25 with 0 score and 25 with 1 score
    # random_0 = all_constraints_scores_flatten[all_constraints_scores_flatten["binary_score"] == 0].sample(25)
    # random_1 = all_constraints_scores_flatten[all_constraints_scores_flatten["binary_score"] == 1].sample(25)
    # random_sample = pd.concat([random_0, random_1]).sample(frac=1)
    # list_for_validation = random_sample[["orig_task", "constraint", "binary_score", "model"]]
    # model_response = []
    # list_dir_generated = list(os.listdir(PATH_TO_GENERATIONS))
    # for _, row in list_for_validation.iterrows():
    #     task = row["orig_task"]
    #     model = row["model"]
    #     for path in list_dir_generated:
    #         if model.lower() in path.lower():
    #             path = os.path.join(PATH_TO_GENERATIONS, path)
    #             break
    #     with open(path, "r") as f:
    #         response = json.load(f)[task]
    #     model_response.append(response[-1]["content"])
    # list_for_validation["response"] = model_response
    # list_for_validation.to_csv(os.path.join("_output", "validation_set.csv"), index=False)
    # load embeddings
    constraints_for_embed = []
    categories_for_embed = []
    embeddings_for_plot = []
    for dump in ["train", "validation", "test"]:
        list_path = CONSTRAINTS_LIST.format(dump=dump)
        embeddings_path = CONSTRAINTS_EMBEDDINGS.format(dump=dump)
        with open(list_path, 'r') as f:
            constraints_list = json.load(f)
        embeddings = np.load(embeddings_path)
        for i, const in enumerate(constraints_list):
            if const in all_unique_constraints and const not in constraints_for_embed:
                category = constraint_to_category[const]
                if category == "Other":
                    continue
                constraints_for_embed.append(const)
                categories_for_embed.append(constraint_to_category[const])
                embeddings_for_plot.append(embeddings[i])
    # plot the embeddings
    tsne = TSNE(n_components=2, random_state=0)
    # sample 1000 points
    np.random.seed(0)
    random_points = np.random.choice(range(len(embeddings_for_plot)), 1000, replace=False)
    embeddings_for_plot = [embeddings_for_plot[i] for i in random_points]
    categories_for_embed = [categories_for_embed[i] for i in random_points]
    embeddings_for_plot = np.array(embeddings_for_plot)
    embeddings_2d = tsne.fit_transform(embeddings_for_plot)
    plt.figure("tsne of embeddings", figsize=(6,6))
    categories_sorted = sorted(set(categories_for_embed))
    colors_for_categories = {c: plt.cm.tab20.colors[i*2] for i, c in enumerate(categories_sorted)}
    colors_for_categories["Other"] = plt.cm.tab20.colors[len(categories_sorted)*2]
    for i, category in enumerate(categories_sorted):
        indices = [j for j, cat in enumerate(categories_for_embed) if cat == category]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=category,
                    color=colors_for_categories[category], alpha=0.9)
    # set legend above plot
    plt.legend(loc='upper left', bbox_to_anchor=(-0.15, -0.05), ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join("_output", "tsne_of_embeddings.png"))

    def sort_key(word):
        first_letter = word[0]  # First letter
        try:
            middle_letter = float(word.split('-')[-1][:-1])  # Middle letter
        except ValueError:
            middle_letter = 0
        return first_letter, middle_letter  # Sorting by first, then middle letter

    models_sorted = sorted(all_scores.keys(), key=sort_key)
    models_sorted.reverse()

    model_colors = [color_map_by_model[model] for model in models_sorted]

    all_domains_scores = []
    domains_for_label = []
    for domain in ordered_categories:
        domain_size = domain_count[domain] / len(ds)
        if domain_size < 0.02:
            continue
        domains_for_label.append(domain)
        domain_scores = []
        for model in models_sorted:
            mean_task_score = all_scores[model].groupby("domain").get_group(domain)["mean_score"].mean()
            domain_scores.append(mean_task_score)
            print(f"{model} mean score for {domain}: {mean_task_score}")
        all_domains_scores.append(domain_scores)
    all_domains_scores = np.array(all_domains_scores)

    # Bar width
    bar_width = 0.05
    x = np.arange(len(domains_for_label))  # Group positions

    # Plot
    fig, ax = plt.subplots(figsize=(24, 6))

    for i, model in enumerate(models_sorted):
        ax.bar(x + i * bar_width, all_domains_scores[:, i], width=bar_width, label=model, color=model_colors[i])

    # Formatting
    ax.set_xticks(x + (bar_width * (len(models_sorted) - 1)) / 2)
    ax.set_xticklabels(domains_for_label, rotation=45, ha='right')
    ax.set_ylabel('Mean task score')
    # reverse the order of labels in legend
    handles, labels_for_legend = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels_for_legend[::-1], bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(os.path.join("_output", "bar_plot_of_mean_score_by_domain.png"))

    plt.figure("frequency of categories", figsize=(6, 6))
    # count how many constraints are in each category
    count = Counter(constraint_to_category.values())
    category_constraint_labels = list(count.keys())
    category_constraint_labels = sorted(category_constraint_labels, key=lambda x: count[x], reverse=True)
    # plot frequencies
    plt.bar(category_constraint_labels, [count[l] for l in category_constraint_labels],
            color=[colors_for_categories[cat] for cat in category_constraint_labels], edgecolor='black', alpha=0.9)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("_output", "frequency_of_categories.png"))

    for i in range(1, 9):
        distribution = Counter(all_constraints_scores["Deepseek-v3"].groupby('total_constraints').get_group(i)['category'].to_list())
        plt.figure(f"frequency of categories_{i}", figsize=(6, 6))
        # count how many constraints are in each category
        # plot frequencies
        plt.bar(category_constraint_labels, [distribution[l] for l in category_constraint_labels],
                color=[colors_for_categories[cat] for cat in category_constraint_labels], edgecolor='black', alpha=0.9)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title(f"{i} Constraints in a task", fontdict={'fontsize': 20})
        plt.tight_layout()
        plt.savefig(os.path.join("_output", f"frequency_of_categories_{i}.png"))


    category_constraint_labels = category_constraint_labels[:-1]

    # heatmap of co-occurances in task level
    heatmap = np.zeros((len(category_constraint_labels), len(category_constraint_labels)))
    heatmap_tasks = np.zeros((len(domains_for_label), len(category_constraint_labels)))
    n_tasks_for_constraints = 0
    n_task_for_domains = 0
    print("calculating heatmap of co-occurrences")
    for i, constraints in tqdm(enumerate(ds["decomposition"]), total=len(ds)):
        categories = set()
        for constraint in constraints:
            constraint_label = constraint_to_category[constraint]
            if constraint_label == "Other":
                continue
            categories.add(constraint_label)
        if len(categories) >= 2:
            n_tasks_for_constraints += 1
            all_pairs = set([(i, j) for i in categories for j in categories])
            for k, j in all_pairs:
                heatmap[category_constraint_labels.index(k), category_constraint_labels.index(j)] += 1
        task_domain = domain_df[domain_df["task"] == ds["task"][i]]["domain"].values[0]
        if task_domain in domains_for_label:
            n_task_for_domains += 1
            for category in categories:
                heatmap_tasks[domains_for_label.index(task_domain), category_constraint_labels.index(category)] += 1

    # normalize heatmap by category frequency
    new_heatmap = np.zeros((len(category_constraint_labels), len(category_constraint_labels)))
    for i in range(len(category_constraint_labels)):
        for j in range(len(category_constraint_labels)):
            if i == j:
                new_heatmap[i, j] = 1
            else:
                new_heatmap[i, j] = heatmap[i, j] / (heatmap[i, i] * heatmap[j, j] / n_tasks_for_constraints)

    # normalize heatmap by category frequency
    new_heatmap_domains = np.zeros((len(domains_for_label), len(category_constraint_labels)))
    for i in range(len(domains_for_label)):
        normalize_factor_i = np.sum(heatmap_tasks[i])
        for j in range(len(category_constraint_labels)):
            normalize_factor_j = np.sum(heatmap_tasks[:, j])
            new_heatmap_domains[i, j] = heatmap_tasks[i, j] / (normalize_factor_i * normalize_factor_j / n_task_for_domains)

    plt.figure("occurrences_domains_and_constraints", figsize=(9, 8))
    img = plt.imshow(new_heatmap_domains, cmap='PiYG', interpolation='nearest', vmin=0, vmax=2, aspect='auto')
    plt.xticks(range(len(category_constraint_labels)), category_constraint_labels, rotation=45, ha='right', fontsize=10)
    plt.xlabel("Constraint category", fontsize=12)
    plt.yticks(range(len(domains_for_label)), domains_for_label, fontsize=10)
    plt.ylabel("Task domain", fontsize=12)
    cbar = plt.colorbar(img)
    cbar.set_label("Observed to expected ratio", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("_output", "heatmap_of_co-occurrences_domains_and_constraints.png"))
    plt.show()

    plt.figure("heatmap of co-occurrences in tasks")

    mask = np.triu(np.ones_like(heatmap, dtype=bool), k=1)
    masked_heatmap = np.ma.masked_where(mask, new_heatmap)

    # Plot the heatmap
    img = plt.imshow(masked_heatmap, cmap='BrBG', interpolation='nearest', vmin=0, vmax=2)

    # Set axis labels
    plt.xticks(range(len(category_constraint_labels)), category_constraint_labels, rotation=45, ha='right')
    plt.yticks(range(len(category_constraint_labels)), category_constraint_labels)

    # Add colorbar and set label
    cbar = plt.colorbar(img)
    cbar.set_label("Observed to expected ratio", fontsize=12)  # Add title to the colorbar

    plt.tight_layout()
    plt.savefig(os.path.join("_output", "heatmap_of_co-occurrences_in_tasks.png"))

    plt.figure("line plot normalized")
    score_1 = {}
    means = []
    for i, model in enumerate(models_sorted):
        df = all_scores[model]
        all_means = all_constraints_scores[model].groupby(["total_constraints", "category"])["score"].mean()
        mean_score_by_num_constraint = []
        for num_constraints in range(1, 9):
            mean_data = all_means.get(num_constraints).mean()
            mean_score_by_num_constraint.append(mean_data)
        means.append(df["mean_score"].mean())
        group_by_num_constraints = df.groupby("num_constraints")["mean_score"].mean()
        # sort by num constraints
        # mean_score_by_num_constraint = group_by_num_constraints.sort_index()
        # score_1[model] = mean_score_by_num_constraint.to_list()[0]
        score_1[model] = mean_score_by_num_constraint[0]
        # plot line plot
        plt.plot(range(1,9), mean_score_by_num_constraint, label=model, color=model_colors[i])
    plt.xlabel("Number of constraints in a task", fontsize=12)
    plt.ylabel("Fraction of fulfilled constraints (normalized)", fontsize=12)
    plt.xticks(range(1, 9))
    # sort the order in legend
    handles, labels_for_figure = plt.gca().get_legend_handles_labels()
    custom_order = sorted(score_1.keys(), key=lambda x: -score_1[x])

    # Sort by custom order
    sorted_handles_labels = sorted(zip(handles, labels_for_figure), key=lambda x: custom_order.index(x[1]))

    # Unzip into sorted handles and labels
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)

    # Add legend
    plt.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1))
    # plt.title(f"score by num constraints")
    plt.tight_layout()
    plt.savefig(os.path.join("_output", "line_plot_of_mean_score_by_num_constraints_normalized.png"))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(models_sorted[::-1], means[::-1], color=model_colors[::-1], width=0.5)

    # Add text labels with the values
    for bar in bars:
        height = bar.get_height()  # Get the height of each bar (corresponding to the value)
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Center text horizontally
            height + 0.01,  # Offset text slightly above the bar
            f"{height:.2f}",  # Format the value to 2 decimal places
            ha='center',  # Align horizontally
            va='bottom'  # Align vertically
        )

    plt.ylabel("Mean fraction of fulfilled constraints", fontsize=12)
    plt.ylim(0, 0.75)
    plt.xticks(rotation=45, ha='right')  # Rotate labels if needed
    plt.tight_layout()
    plt.savefig(os.path.join("_output", "bar_plot_of_mean_score.png"))

    print("\n\nMean scores:")
    for model in all_scores:
        print(f"{model}: {all_scores[model]['mean_score'].mean().round(3)}")

    plt.figure("bar plot of mean score by category")
    all_cat_scores = []
    for category in category_constraint_labels:
        cat_scores = []
        constraints_in_cat = [constraint for constraint in constraint_to_category if constraint_to_category[constraint] == category]
        for model in models_sorted:
            mean_category_score = all_constraints_scores[model].loc[all_constraints_scores[model]["constraint"].isin(constraints_in_cat)]["score"].mean()
            cat_scores.append(mean_category_score)
            print(f"{model} mean score for {category}: {mean_category_score}")
        all_cat_scores.append(cat_scores)
    all_cat_scores = np.array(all_cat_scores)

    # Bar width
    bar_width = 0.05
    x = np.arange(len(category_constraint_labels))  # Group positions

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models_sorted):
        ax.bar(x + i * bar_width, all_cat_scores[:, i], width=bar_width, label=model, color=model_colors[i])

    # Formatting
    ax.set_xticks(x + (bar_width * (len(models_sorted) - 1)) / 2)
    ax.set_xticklabels(category_constraint_labels, rotation=45, ha='right')
    ax.set_ylabel('Fraction of fulfilled constraints')
    # reverse the order of labels in legend
    handles, labels_for_legend = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels_for_legend[::-1], bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(os.path.join("_output", "bar_plot_of_mean_score_by_category.png"))

    plt.figure("bar plot of mean score by category only big models")
    bar_width = 0.12
    models_to_include = ["Deepseek-v3", "Gemma-2-9b", "Llama3.3-70b", "Llama3.1-405b",  "Mistral-large",  "Qwen2.5-72b"]
    only_big_model_scores = all_cat_scores[:, [models_sorted.index(model) for model in models_to_include]]
    only_big_model_colors = [model_colors[models_sorted.index(model)] for model in models_to_include]
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models_to_include):
        ax.bar(x + i * bar_width, only_big_model_scores[:, i], width=bar_width, label=model, color=only_big_model_colors[i])

    # Formatting
    ax.set_xticks(x + (bar_width * (len(models_to_include) - 1)) / 2)
    ax.set_xticklabels(category_constraint_labels, rotation=45, ha='right')
    ax.set_ylabel('Fraction of fulfilled constraints')
    ax.legend(bbox_to_anchor=(0, 1.1), loc='upper left', ncol=len(models_to_include))
    plt.tight_layout()

    plt.savefig(os.path.join("_output", "bar_plot_of_mean_score_by_category_only_big_models.png"))


    for i, model in enumerate(models_sorted):
        # plot bar plot for each model
        plt.figure(f"{model} bar plot of mean score by category")
        model_scores = all_cat_scores[:, i]
        plt.bar(category_constraint_labels, model_scores, color=model_colors[i], edgecolor='black')
        plt.ylim(0, 0.8)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Fraction of fulfilled constraints')
        plt.title(f"Mean score by category for {model}")
        plt.tight_layout()
        plt.savefig(os.path.join("_output", f"bar_plot_of_mean_score_by_category_{model}.png"))

    pairs = {}
    mean_category_scores = {}
    for model in models_sorted:
        plt.figure(f"lineplot of mean score by category by num constraints {model}")
        model_constraint_scores = all_constraints_scores[model]
        group_by_cat_and_num_constraints = model_constraint_scores.groupby(["category", "total_constraints"])
        for category in category_constraint_labels:
            scores = []
            for num_constraints in range(1, 9):
                try:
                    mean_score = group_by_cat_and_num_constraints.get_group((category, num_constraints))["score"].mean()
                except KeyError:
                    print(f"no score for {category} with {num_constraints} constraints")
                    mean_score = -999
                scores.append(mean_score)
            while scores[-1] == -999:
                scores.pop()
            plt.plot(scores, label=category)
        plt.xticks(range(8), [str(i) for i in range(1, 9)])
        plt.xlabel("Number of constraints in a task", fontsize=12)
        plt.ylabel("Mean fraction of fulfilled constraints", fontsize=12)
        plt.ylim(0, 1)
        plt.title(f"Mean score by category by num constraints for {model}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("_output", f"lineplot_of_mean_score_by_category_by_num_constraints_{model}.png"))

        group_by_orig_task = model_constraint_scores.groupby("orig_task")
        for task in tqdm(ds["task"]):
            task_rows = group_by_orig_task.get_group(task)
            if len(task_rows) != 2:
                continue
            cat = task_rows["category"].to_list()
            scores = task_rows["score"].to_list()
            if cat[0] not in category_constraint_labels or cat[1] not in category_constraint_labels:
                continue
            if (cat[0], cat[1]) not in pairs:
                pairs[(cat[0], cat[1])] = []
                mean_category_scores[cat[0]] = []
                if cat[1] != cat[0]:
                    pairs[(cat[1], cat[0])] = []
                    mean_category_scores[cat[1]] = []
            mean_category_scores[cat[0]].append(scores[0])
            mean_category_scores[cat[1]].append(scores[1])
            pairs[(cat[0], cat[1])].append(scores[0])
            pairs[(cat[1], cat[0])].append(scores[1])
    for cat in mean_category_scores:
        mean_category_scores[cat] = np.mean(mean_category_scores[cat])
    heatmap = np.zeros((len(category_constraint_labels), len(category_constraint_labels)))

    for cat1, cat2 in pairs:
        # normalized_score = pairs[(cat1, cat2)] / mean_category_scores[cat1]
        heatmap[category_constraint_labels.index(cat1), category_constraint_labels.index(cat2)] = np.mean(pairs[(cat1, cat2)]) / mean_category_scores[cat1]
    # save the heatmap to a csv
    np.savetxt(os.path.join("_output", "heatmap_of_scores_co-occurrences_in_tasks_mean_all_models.csv"), heatmap, delimiter=",")
    plt.figure(f"heatmap of constraint scores co-occurrences in tasks - mean all models")
    plt.imshow(heatmap, cmap='PRGn', interpolation='nearest')
    plt.clim(0, 2)
    plt.xticks(range(len(category_constraint_labels)), category_constraint_labels, rotation=45, ha='right')
    plt.yticks(range(len(category_constraint_labels)), category_constraint_labels)
    cb = plt.colorbar()
    cb.set_label("Ratio to mean performance", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("_output", f"heatmap_of_scores_co-occurrences_in_tasks_mean_all_models.png"))


def get_all_scores(all_jsons, token_count, ds, domain_df):
    all_scores = {k: [] for k in all_jsons}
    all_constraints_scores = {k: [] for k in all_jsons}
    df_categories = pd.read_csv(CSV_PATH)
    dict_constraints = {}
    # for each constraint, map to a category
    for _, row in df_categories.iterrows():
        category = row["categories"]
        other = row["Other"]
        if other == 1 or pd.isna(category):
            category = "Other"
        dict_constraints[row["constraint"]] = category
    blacklist_ids = []
    for k in all_jsons:
        for i, row in enumerate(ds):
            task = row["task"]
            decomposition_len = len(row["decomposition"])
            num_constraints = len(all_jsons[k][task]["scores"])
            domain = domain_df[domain_df["task"] == task]["domain"].values[0]
            if pd.isna(domain):
                domain = "Other"
            if num_constraints > 8:
                continue
            scores = []
            for constraint in all_jsons[k][task]["scores"]:
                s = all_jsons[k][task]["scores"][constraint]
                if s == 'ERR':
                    s = 0
                pos_category = dict_constraints[constraint] if constraint in dict_constraints else 'Other'
                all_constraints_scores[k].append({"constraint": constraint, "category": pos_category,
                                                  "score": s, "total_constraints": num_constraints, "orig_task": task})
                scores.append(s)
            binary_scores = [1 if s >= 0.5 else 0 for s in scores]
            if len(binary_scores) == 0:
                print(f"no scores for {task}")
                blacklist_ids.append(row["conversation_id"])
                continue
            mean_score = np.mean(binary_scores)
            num_constraints = len(binary_scores)
            all_scores[k].append((mean_score, num_constraints, token_count[i], decomposition_len, domain))
        all_scores[k] = pd.DataFrame(all_scores[k], columns=["mean_score", "num_constraints", "token_count", "decomposition_len", "domain"])
        not_matching = np.where(all_scores[k]["num_constraints"] != all_scores[k]["decomposition_len"])[0]
        blacklist_ids.extend(ds.select(not_matching)["conversation_id"])
        all_constraints_scores[k] = pd.DataFrame(all_constraints_scores[k])
    print(f"blacklist ids:\n{list(set(blacklist_ids))}\n\n")
    return all_scores, all_constraints_scores, dict_constraints


if __name__ == '__main__':
    llama_models = ["Llama-3.1-8B-Instruct", "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"]
    gemma_models = ["gemma-2-2b-it", "gemma-2-9b-it"]
    qwen_models = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"]
    big_models_prefix = "init-generations-via-rits-{model}.{split}.json"
    big_models = ["llama3.3-70b", "mistral-large", "qwen2.5-72b", "llama3.1-405b", "deepseek-v3"]
    all_paths = {}

    # llama_models = ["Llama-3.1-8B-Instruct"]
    # big_models = ["llama3.1-8b"]
    # gemma_models = []
    # qwen_models = []

    for name, models in zip(["Llama", "Gemma", "Qwen"], [llama_models, gemma_models, qwen_models]):
        print(f'\n\n{name}')
        if name == 'Llama' or name == 'Qwen':
            to_remove = "-Instruct"
        else:  # Gemma
            to_remove = "-it"
        init = {model.replace(to_remove, "").capitalize(): PATH.format(step="combined") + model + f"-{DUMP}-init-gen.json"
                for model in models}
        all_paths.update(init)

    for model in big_models:
        path = PATH.format(step="combined") + big_models_prefix.format(model=model, split=DUMP)
        all_paths[model.capitalize()] = path

    calculate(all_paths)



