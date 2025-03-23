from argparse import ArgumentParser
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

np.random.seed(42)


def prepare_labels(path_to_dataset):
    labels_df = pd.read_csv(path_to_dataset)
    labels_df = labels_df[labels_df["gemma_0shot"] < 1]
    labels_df = labels_df[~labels_df["all_three_models_same"]]
    return labels_df


def prepare_ds(path_to_embeddings, path_to_samples_list, path_to_dataset):
    embeddings = np.load(path_to_embeddings)
    with open(path_to_samples_list, "r") as f:
        samples_list = json.load(f)
    labels = prepare_labels(path_to_dataset)

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
        "kNN": KNeighborsClassifier(n_neighbors=20),
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }
    print(Counter(y_test))

    random = np.random.RandomState(42).random_integers(y_test.min(), y_test.max() + 1, len(y_test))
    random_acc = (random == y_test).mean()
    print(f"Random Accuracy: {random_acc:.2f}")

    # shuffle y_test
    y_test_shuffled = np.random.RandomState(42).permutation(y_test)
    shuffled_acc = (y_test_shuffled == y_test).mean()
    print(f"Shuffle Accuracy: {shuffled_acc:.2f}")

    preds = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        preds[name] = y_pred.tolist()
        accuracy = (y_pred == y_test).mean()
        print(Counter(y_pred))
        print(f"{name} Accuracy: {accuracy:.2f}")
    return preds


def main(path_to_embeddings, path_to_samples_list, path_to_labels):
    all_data = prepare_ds(path_to_embeddings, path_to_samples_list, path_to_labels)

    # Split into train and test
    train_data = all_data.sample(frac=0.8, random_state=42)
    test_data = all_data.drop(train_data.index)

    X_train = np.stack(train_data["embedding"].values)
    y_train = train_data["label"]  # Keep full binary format
    X_test = np.stack(test_data["embedding"].values)
    y_test = test_data["label"]

    # Train and evaluate sklearn models
    predictions = train_sklearn_models(X_train, y_train, X_test, y_test)

    # Save predictions
    for name, preds in predictions.items():
        out_json = {}
        for i, sample in enumerate(test_data["sample"]):
            out_json[sample] = predictions[name][i]
        with open(f"_output/simple_classifiers_same_size/{name}.json", "w") as f:
            str_to_write = json.dumps(out_json, indent=2)
            f.write(str_to_write)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--samples_list", required=True)
    parser.add_argument("--labels", required=True)

    args = parser.parse_args()

    main(args.embeddings, args.samples_list, args.labels)
