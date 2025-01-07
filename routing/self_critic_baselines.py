from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from routing.train_self_critic_prediction import set_random_seed, generate_training_data


def main(path_to_config, path_to_df, seed):
    # Set seed for reproducibility
    set_random_seed(seed)

    # Load config and data
    with open(path_to_config, 'r') as f:
        config = json.load(f)
    df = pd.read_csv(path_to_df)
    dataset = generate_training_data(df, config)
    dataset = dataset.shuffle(seed=seed).train_test_split(test_size=0.2)

    labels = dataset['test']['is_self_critic_best']
    print(f"Number of positive labels: {np.sum(labels)}")
    print(f"Number of negative labels: {len(labels) - np.sum(labels)}")
    print("nuber of no critics needed: ", np.sum(dataset['test']['no_critics_needed']))
    all_ones = np.ones_like(labels)
    all_zeros = np.zeros_like(labels)
    shuffled_labeled = np.random.permutation(labels)
    f1_shuffled = f1_score(labels, shuffled_labeled, average='binary')
    f1_all_ones = f1_score(labels, all_ones, average='binary')
    f1_all_zeros = f1_score(labels, all_zeros, average='binary')
    print(f"F1 score for predicting all ones: {f1_all_ones}")
    print(f"F1 score for predicting all zeros: {f1_all_zeros}")
    print(f"F1 score for predicting shuffled labels: {f1_shuffled}")
    accuracy_all_ones = accuracy_score(labels, all_ones)
    accuracy_all_zeros = accuracy_score(labels, all_zeros)
    accuracy_shuffled = accuracy_score(labels, shuffled_labeled)
    print(f"Accuracy for predicting all ones: {accuracy_all_ones}")
    print(f"Accuracy for predicting all zeros: {accuracy_all_zeros}")
    print(f"Accuracy for predicting shuffled labels: {accuracy_shuffled}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--df', type=str, required=True, help='Path to dataframe with labels')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    main(
        args.config,
        args.df,
        args.seed
    )
