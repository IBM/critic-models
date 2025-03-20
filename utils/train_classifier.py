from argparse import ArgumentParser
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

np.random.seed(42)


def prepare_labels(labels_path, generator_model, label_index):
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df[labels_df["generator_model"] == generator_model]
    labels_df = labels_df.drop(columns=["generator_model", "Unnamed: 0"])
    # drop rows with all last three columns == 1
    # labels_df = labels_df[~(labels_df.iloc[:, -3:] == 1).all(axis=1)]
    # set label to 1 if all last three columns == 1
    if label_index == -1:
        labels_df["label"] = (labels_df.iloc[:, -3:] == 1).all(axis=1).astype(int)
    else:
        labels_df = labels_df[~(labels_df.iloc[:, -3:] == 1).all(axis=1)]
        labels_df["label"] = labels_df.iloc[:, -1-label_index]
    print(labels_df["label"].sum()/len(labels_df))
    return labels_df


def prepare_ds(path_to_embeddings, path_to_samples_list, path_to_labels, generator_model, label_index):
    embeddings = np.load(path_to_embeddings)
    with open(path_to_samples_list, "r") as f:
        samples_list = json.load(f)
    labels = prepare_labels(path_to_labels, generator_model, label_index)

    # Filter labels to include only samples in the list
    labels = labels[labels["sample"].isin(samples_list)]
    samples_list = [sample for sample in samples_list if sample in labels["sample"].values]

    # Reorder embeddings to match the filtered labels
    embeddings = np.array([embeddings[samples_list.index(sample)] for sample in labels["sample"].values])

    all_data = labels
    all_data["embedding"] = embeddings.tolist()
    return all_data


def train_sklearn_models(X_train, y_train, X_test, y_test):
    classifiers = {
        "kNN": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "SVM": SVC(kernel="rbf", probability=True, class_weight='balanced'),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                     scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
    }

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train.ravel())
        y_pred = clf.predict(X_test)
        print(sum(y_pred)/len(y_pred))
        accuracy = (y_pred == y_test.ravel()).mean()
        print(f"{name} Accuracy: {accuracy:.2f}")


def main(path_to_embeddings, path_to_samples_list, path_to_labels, generator_model, label_index):
    all_data = prepare_ds(path_to_embeddings, path_to_samples_list, path_to_labels, generator_model, label_index)

    # Split into train and test
    train_data = all_data.sample(frac=0.8, random_state=42)
    test_data = all_data.drop(train_data.index)

    X_train = np.stack(train_data["embedding"].values)
    y_train = train_data["label"].values  # Keep full binary format
    X_test = np.stack(test_data["embedding"].values)
    y_test = test_data["label"].values

    # Train and evaluate sklearn models
    train_sklearn_models(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--samples_list", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--generator_model", required=True)
    parser.add_argument("--score_by_sample", required=True)
    parser.add_argument("--label_index", required=True, type=int, default=-1)

    args = parser.parse_args()

    main(args.embeddings, args.samples_list, args.labels, args.generator_model, args.label_index)
