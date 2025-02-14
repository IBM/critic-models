import os.path
import numpy as np
import json
from datasets import load_dataset
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm

CSV_PATH = "/Users/gililior/research/datasets/arena_data_v2/categories_single.csv"
DOMAINS_PATH = "/Users/gililior/research/datasets/arena_data_v2/domains.csv"

DUMP = "test"

SCORES_DIR = "/Users/gililior/research/datasets/arena_data_v2/llm_aaj/{step}"
SCORES_PREFIX = "decomposition-evaluation-llama3.1-70b."
PATH = os.path.join(SCORES_DIR, SCORES_PREFIX)



def calculate(paths):
    all_jsons = {}
    ds = load_dataset("gililior/wild-if-eval", split="test")
    # filter tasks with more than 8 constraints

    print("number of tasks", len(ds))
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    token_count = [len(tokenizer.encode(text)) for text in ds["task"]]
    all_constraints = ds["decomposition"]
    all_lengths = [len(c) for c in all_constraints]
    # correlation between token count and length of constraints
    print("correlation between token count and length of constraints", np.corrcoef(token_count, all_lengths)[0, 1])
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

    all_scores, all_constraints_scores, constraint_to_category = get_all_scores(all_jsons, token_count, ds)
    plt.figure(figsize=(10, 10))
    sorted_keys = sorted(all_constraints_scores.keys())
    data_for_violin = [all_constraints_scores[k]["score"].values.tolist() for k in sorted_keys]
    plt.violinplot(data_for_violin, showmeans=True, vert=False, widths=2)
    plt.yticks(range(1, len(sorted_keys) + 1), sorted_keys)
    plt.title("Violin plot of constraints scores")
    plt.tight_layout()
    plt.savefig(os.path.join("_output", "violin_plot_of_constraints.png"))

    for k in all_constraints_scores:
        df = all_constraints_scores[k]
        # sort alphabetically
        df = df.sort_values("constraint")
        groups = df.groupby("total_constraints").groups
        one_constraint = df.loc[groups[1]]
        # sort by score
        one_constraint = one_constraint.sort_values("score")
        # plot frequency of constraints - how many constraints appear one time, how many two times, etc...
        plt.figure(f"{k} constraints")
        count = Counter(df["constraint"])
        # plot frequencies
        # how many keys appear 1 time
        counts = Counter(count.values())
        print(counts)


        # count how many equal 0
        print(f"{k} has {len(one_constraint[one_constraint['score'] == 0])} constraints with score 0, out of {len(one_constraint)}")
        # for_violin = [df.loc[groups[num_constraints]]["score"].values.tolist() for num_constraints in groups]
        # plt.figure(f"{k} constraints")
        # plt.violinplot(for_violin, showmeans=True, vert=False, widths=2)
        # plt.yticks(range(1, len(groups) + 1), groups.keys())
        # plt.title(f"Violin plot of constraints scores for {k}")
        # plt.tight_layout()
        # plt.show()

    # plot the distribution of num constraints
    all_models = list(all_scores.keys())

    only_constraints = all_scores[all_models[0]]["num_constraints"].to_list()
    # violin plot
    plt.violinplot(only_constraints, showmeans=True)
    # count frequency of each num constraints
    print(Counter(only_constraints))

    plt.figure("histogram of num constraints")
    plt.bar(range(1, 9), [only_constraints.count(i) for i in range(1, 9)], color='#ff7f0e', edgecolor='black')
    # plt.hist(only_constraints, alpha=0.5, color='b', edgecolor='black')
    # print(set(only_constraints))
    plt.xticks(range(1, 9), [f"{i}" for i in range(1, 9)])
    plt.ylabel("Tasks count", fontsize=12)
    plt.xlabel("Constraints per task", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("_output", "histogram_of_num_constraints.png"))

    plt.figure("line plot")

    # generate color list in length of all_scores, that colors are different
    colors = [plt.cm.tab20(i) for i in range(20)]
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]

    score_1 = {}
    means = []

    def sort_key(word):
        first_letter = word[0]  # First letter
        try:
            middle_letter = float(word.split('-')[-1][:-1])  # Middle letter
        except ValueError:
            middle_letter = 0
        return first_letter, middle_letter  # Sorting by first, then middle letter

    models_sorted = sorted(all_scores.keys(), key=sort_key)
    models_sorted.reverse()
    for i, model in enumerate(models_sorted):
        df = all_scores[model]
        means.append(df["mean_score"].mean())
        group_by_num_constraints = df.groupby("num_constraints").mean()
        # sort by num constraints
        group_by_num_constraints = group_by_num_constraints.sort_values("num_constraints")
        score_1[model] = group_by_num_constraints["mean_score"].to_list()[0]
        # plot line plot
        plt.plot(group_by_num_constraints.index.to_list(), group_by_num_constraints["mean_score"], label=model, color=hex_colors[i])
    plt.xlabel("Number of constraints in a task", fontsize=12)
    plt.ylabel("Mean fraction of fulfilled constraints", fontsize=12)
    plt.xticks(range(1, 9))
    # sort the order in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    custom_order = sorted(score_1.keys(), key=lambda x: -score_1[x])

    # Sort by custom order
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: custom_order.index(x[1]))

    # Unzip into sorted handles and labels
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)

    # Add legend
    plt.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1))
    # plt.title(f"score by num constraints")
    plt.tight_layout()
    plt.savefig(os.path.join("_output", "line_plot_of_mean_score_by_num_constraints.png"))

    for i, model in enumerate(models_sorted):
        # correlation between token count and mean score
        plt.figure(f"{model} scatter plot")
        plt.scatter(all_scores[model]["token_count"], all_scores[model]["mean_score"], color=hex_colors[i])
        plt.xlabel("Token count", fontsize=12)
        plt.ylabel("Mean fraction of fulfilled constraints", fontsize=12)
        plt.title(f"Correlation between token count and mean score for {model}")
        plt.tight_layout()
        plt.savefig(os.path.join("_output", f"scatter_plot_{model}.png"))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(models_sorted[::-1], means[::-1], color=hex_colors[::-1], width=0.5)

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

    plt.figure("frequency of categories")
    # count how many constraints are in each category
    count = Counter(constraint_to_category.values())
    labels = list(count.keys())
    # plot frequencies
    plt.bar(labels, [count[l] for l in labels], color='#9467bd', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("_output", "frequency_of_categories.png"))

    plt.figure("bar plot of mean score by category")

    all_cat_scores = []
    labels = labels[:-1]
    for category in labels:
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
    x = np.arange(len(labels))  # Group positions

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models_sorted):
        ax.bar(x + i * bar_width, all_cat_scores[:, i], width=bar_width, label=model, color=hex_colors[i])

    # Formatting
    ax.set_xticks(x + (bar_width * (len(models_sorted) - 1)) / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Fraction of fulfilled constraints')
    # reverse the order of labels in legend
    handles, labels_for_legend = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels_for_legend[::-1], bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(os.path.join("_output", "bar_plot_of_mean_score_by_category.png"))

    for i, model in enumerate(models_sorted):
        # plot bar plot for each model
        plt.figure(f"{model} bar plot of mean score by category")
        model_scores = all_cat_scores[:, i]
        plt.bar(labels, model_scores, color=hex_colors[i], edgecolor='black')
        plt.ylim(0, 0.8)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Fraction of fulfilled constraints')
        plt.title(f"Mean score by category for {model}")
        plt.tight_layout()
        plt.savefig(os.path.join("_output", f"bar_plot_of_mean_score_by_category_{model}.png"))

    # heatmap of co-occurances in task level
    heatmap = np.zeros((len(labels), len(labels)))
    heatmap_tasks = np.zeros((len(ordered_categories), len(labels)))
    n_tasks = 0
    for i, constraints in enumerate(ds["decomposition"]):
        task_domain = domain_df[domain_df["task"] == ds["task"][i]]["domain"].values[0]
        categories = set()
        for constraint in constraints:
            constraint_label = constraint_to_category[constraint]
            if constraint_label == "Other":
                continue
            categories.add(constraint_label)
        if len(categories) < 2:
            continue
        n_tasks += 1
        all_pairs = set([(i, j) for i in categories for j in categories])
        for k, j in all_pairs:
            heatmap[labels.index(k), labels.index(j)] += 1
        # if task_domain == "Artificial Intelligence" or pd.isna(task_domain):
        #     continue
        # for category in categories:
        #     heatmap_tasks[ordered_categories.index(task_domain), labels.index(category)] += 1
    # normalize heatmap by category frequency
    new_heatmap = np.zeros((len(labels), len(labels)))
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                new_heatmap[i, j] = 1
            else:
                new_heatmap[i, j] = heatmap[i, j] / (heatmap[i, i] * heatmap[j, j] / n_tasks)


    # heatmap_tasks /= heatmap_tasks.sum(axis=1)[:, np.newaxis]
    #
    #
    # plt.figure("heatmap of co-occurrences  domains_and_constraints")
    #
    # # heatmap of coocurances between tasks and constraints
    # img = plt.imshow(heatmap_tasks, cmap='Reds', interpolation='nearest')
    #
    # # Set axis labels
    # plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    # plt.yticks(range(len(ordered_categories)), ordered_categories)
    #
    # # Add colorbar and set label
    # cbar = plt.colorbar(img)
    # # cbar.set_label("% tasks", fontsize=12)  # Add title to the colorbar
    #
    # plt.tight_layout()
    # plt.savefig(os.path.join("_output", "heatmap_of_co-occurrences_domains_and_constraints.png"))

    plt.figure("heatmap of co-occurrences in tasks")

    mask = np.triu(np.ones_like(heatmap, dtype=bool), k=1)
    masked_heatmap = np.ma.masked_where(mask, new_heatmap)

    # Plot the heatmap
    img = plt.imshow(masked_heatmap, cmap='BrBG', interpolation='nearest', vmin=0, vmax=2)

    # Set axis labels
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.yticks(range(len(labels)), labels)

    # Add colorbar and set label
    cbar = plt.colorbar(img)
    cbar.set_label("Expected ratio", fontsize=12)  # Add title to the colorbar

    plt.tight_layout()
    plt.savefig(os.path.join("_output", "heatmap_of_co-occurrences_in_tasks.png"))

    pairs = {}
    each_cat_mean = np.zeros(len(labels))
    for model in models_sorted:
        plt.figure(f"lineplot of mean score by category by num constraints {model}")
        model_constraint_scores = all_constraints_scores[model]
        group_by_cat_and_num_constraints = model_constraint_scores.groupby(["category", "total_constraints"])
        for category in labels:
            mean_cat = model_constraint_scores[model_constraint_scores["category"] == category]["score"].mean()
            each_cat_mean[labels.index(category)] += mean_cat
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
        plt.ylim(0, 0.8)
        plt.title(f"Mean score by category by num constraints for {model}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("_output", f"lineplot_of_mean_score_by_category_by_num_constraints_{model}.png"))

        for task in tqdm(ds["task"]):
            task_rows = model_constraint_scores[model_constraint_scores["orig_task"] == task]
            cat = task_rows["category"].to_list()
            scores = task_rows["score"].to_list()
            for i, category in enumerate(cat):
                for j, other_category in enumerate(cat):
                    if i == j:
                        continue
                    if (category, other_category) not in pairs:
                        pairs[(category, other_category)] = []
                    pairs[(category, other_category)].append(scores[i])
    for pair in pairs:
        pairs[pair] = np.mean(pairs[pair])
    heatmap = np.zeros((len(labels), len(labels)))
    for cat1, cat2 in pairs:
        heatmap[labels.index(cat1), labels.index(cat2)] = pairs[(cat1, cat2)]
    each_cat_mean /= len(models_sorted)
    # concatenate the mean of each category to the heatmap as another column
    heatmap = np.concatenate((heatmap, each_cat_mean.reshape(-1, 1)), axis=1)
    # save the heatmap to a csv
    np.savetxt(os.path.join("_output", "heatmap_of_scores_co-occurrences_in_tasks_mean_all_models.csv"), heatmap, delimiter=",")
    plt.figure(f"heatmap of constraint scores co-occurrences in tasks - mean all models")
    plt.imshow(heatmap, cmap='Blues', interpolation='nearest')
    plt.xticks(range(len(labels)+1), labels + ['Baseline'], rotation=45, ha='right')
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join("_output", f"heatmap_of_scores_co-occurrences_in_tasks_mean_all_models.png"))













def get_all_scores(all_jsons, token_count, ds):
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
            all_scores[k].append((mean_score, num_constraints, token_count[i], decomposition_len))
        all_scores[k] = pd.DataFrame(all_scores[k], columns=["mean_score", "num_constraints", "token_count", "decomposition_len"])
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



