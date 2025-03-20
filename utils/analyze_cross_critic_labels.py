from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def analyze_labels(path_to_labels):
    labels_df = pd.read_csv(path_to_labels)
    if "best_gen_models" in labels_df.columns:
        best_model_col = "best_gen_models"
        class_model_col = "revise_model"
    else:
        best_model_col = "best_rev_models"
        class_model_col = "generator_model"
    all_models = labels_df[class_model_col].unique()
    # add column for each model, to be indicator if it appears in best model col list
    for model in all_models:
        labels_df[model] = labels_df[best_model_col].apply(lambda x: model in x)
    # drop the best model col
    labels_df = labels_df.drop(columns=[best_model_col])
    labels_df.to_csv(path_to_labels.replace('.csv', '_for_training.csv'), index=False)

    print_distribution(all_models, best_model_col, class_model_col, labels_df)

    # remove rows with more than one model in the best model col list
    labels_reduced = labels_df[labels_df[all_models].sum(axis=1) < len(all_models)]
    all_models_are_best = labels_df[labels_df[all_models].sum(axis=1) == len(all_models)]
    # plot the distribution of scores for the samples where all models are in the best model col list
    plt.violinplot(all_models_are_best["best_rev_score"])
    plt.title(f"Distribution of scores for samples where\nall models provided the same revision\n{path_to_labels}")
    plt.show()
    print("Number of total samples:", labels_df.shape[0])
    print("Number of unique samples:", labels_df["sample"].unique().shape[0])
    print(f"Number of samples with not all models in the best model col list: {labels_reduced.shape[0]}")
    group_by_class_model = labels_reduced.groupby(class_model_col)
    for class_model in group_by_class_model:
        print(f"Number of samples with {class_model[0]}: {class_model[1].shape[0]}")
    print_distribution(all_models, best_model_col, class_model_col, labels_reduced)





def print_distribution(all_models, best_model_col, class_model_col, labels_df):
    # for each model, calculate the number of times it appears in the best model col list
    num_rows = labels_df.shape[0]
    model_counts = {model: np.round(np.sum(labels_df[model]) / num_rows * 100, 1).item() for model in all_models}
    print(f"For {best_model_col}, counts are", model_counts)
    group_by_class_model = labels_df.groupby(class_model_col)
    # for each model, calculate the number of times it appears in the best model col list, grouped by class model
    for class_model in group_by_class_model:
        num_rows = len(class_model[1])
        counts_best_models = {model: np.round(np.sum(class_model[1][model]) / num_rows * 100, 1).item() for model in
                              all_models}
        print("For", class_model_col, class_model[0], "counts are", counts_best_models)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()
    analyze_labels(args.labels)