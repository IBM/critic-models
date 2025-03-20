from argparse import ArgumentParser
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)


def prepare_labels(labels_path, generator_model):
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df[labels_df["generator_model"] == generator_model]
    labels_df = labels_df.drop(columns=["generator_model", "Unnamed: 0"])
    # drop rows with all last three columns == 1
    labels_df = labels_df[~(labels_df.iloc[:, -3:] == 1).all(axis=1)]
    return labels_df


def prepare_ds(path_to_embeddings, path_to_samples_list, path_to_labels, generator_model):
    embeddings = np.load(path_to_embeddings)
    with open(path_to_samples_list, "r") as f:
        samples_list = json.load(f)
    labels = prepare_labels(path_to_labels, generator_model)

    # Filter labels to include only samples in the list
    labels = labels[labels["sample"].isin(samples_list)]
    samples_list = [sample for sample in samples_list if sample in labels["sample"].values]

    # Reorder embeddings to match the filtered labels
    embeddings = np.array([embeddings[samples_list.index(sample)] for sample in labels["sample"].values])

    all_data = labels
    all_data["embedding"] = embeddings.tolist()
    return all_data


class MLPWithSigmoid(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3):
        super(MLPWithSigmoid, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)


def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.float(), labels.float()
            optimizer.zero_grad()

            outputs = model(embeddings).squeeze()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    labels_rev_scores = []
    predicted_rev_scores = []
    random_rev_scores = []
    accuracies = []
    random_accuracies = []
    with torch.no_grad():
        for embeddings, labels, rev_scores in test_loader:
            embeddings, labels = embeddings.float(), labels.float()
            outputs = model(embeddings).squeeze()
            predicted_best = outputs.argmax(dim=1)
            labels_best = labels.argmax(dim=1)
            for i in range(len(labels)):
                labels_rev_scores.append(rev_scores[i, labels_best[i]])
                predicted_rev_scores.append(rev_scores[i, predicted_best[i]])
                accuracies.append(rev_scores[i, predicted_best[i]] == rev_scores[i, labels_best[i]])
                random_index = np.random.randint(0, labels.shape[1])
                random_rev_scores.append(rev_scores[i, random_index])
                random_accuracies.append(rev_scores[i, random_index] == rev_scores[i, labels_best[i]])
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    print(f"Test Loss: {total_loss:.4f}")
    print("Predicted rev scores:", np.mean(predicted_rev_scores))
    print("Gold rev scores:", np.mean(labels_rev_scores))
    print("Random rev scores:", np.mean(random_rev_scores))
    print("Accuracy:", np.mean(accuracies))
    print("Random Accuracy:", np.mean(random_accuracies))


def main(path_to_embeddings, path_to_samples_list, path_to_labels, generator_model, path_to_score_by_sample):
    all_data = prepare_ds(path_to_embeddings, path_to_samples_list, path_to_labels, generator_model)
    score_by_sample = pd.read_csv(path_to_score_by_sample)
    score_by_sample = score_by_sample[score_by_sample["generator_model"] == generator_model]

    # Split into train and test
    train_data = all_data.sample(frac=0.8, random_state=42)
    test_data = all_data.drop(train_data.index)

    X_train = np.stack(train_data["embedding"].values)
    y_train = train_data.drop(columns=["embedding", "best_rev_score", "sample"]).values  # Keep full multi-label format
    X_test = np.stack(test_data["embedding"].values)
    y_test = test_data.drop(columns=["embedding", "best_rev_score", "sample"]).values

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))

    models_sorted = train_data.drop(columns=["embedding", "best_rev_score", "sample"]).columns
    samples_test_sorted = test_data["sample"].values
    rev_scores_sorted_test = []
    for sample in samples_test_sorted:
        current_sample = []
        current_sample_df = score_by_sample[score_by_sample["sample"] == sample]
        current_sample_df = current_sample_df[current_sample_df["generator_model"] == generator_model]
        for rev_model in models_sorted:
            current_sample.append(current_sample_df[current_sample_df["revise_model"] == rev_model]["score"].values[0])
        rev_scores_sorted_test.append(current_sample)
    rev_scores_sorted_test = np.array(rev_scores_sorted_test)
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test), torch.tensor(rev_scores_sorted_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = X_train.shape[1]
    hidden_dim = 512
    output_dim = y_train.shape[1]  # Ensure correct output size

    model = MLPWithSigmoid(input_dim, hidden_dim, output_dim)
    criterion = nn.BCELoss()  # Binary cross-entropy for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    print("Training Model...")
    train_model(model, train_loader, criterion, optimizer, epochs=30)

    print("Evaluating Model...")
    evaluate_model(model, test_loader, criterion)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--samples_list", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--generator_model", required=True)
    parser.add_argument("--score_by_sample", required=True)
    args = parser.parse_args()

    main(args.embeddings, args.samples_list, args.labels, args.generator_model, args.score_by_sample)