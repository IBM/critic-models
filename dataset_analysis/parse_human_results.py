import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np
import krippendorff
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss


all_dfs = []
for path in os.listdir("results"):
    name = path.split("_")[-1].replace(".csv", "")
    df = pd.read_csv(os.path.join("results", path))
    df["Name"] = name
    all_dfs.append(df)

all_results = pd.concat(all_dfs)

print("number of unique tasks:", len(all_results["Task"].unique()))
print("total of number of annotations:", len(all_results))
print("number of tasks with more than one annotation:", len(all_results["Task"].value_counts()[all_results["Task"].value_counts() > 1]))
print("number of annotations for each worker:", all_results["Name"].value_counts())

for category in ["Correctness", "Completeness", "Independence"]:
    print(f"Average {category} rating:", all_results[category].mean())


# get all pairs of annotators
pairs = []
# unique_names = all_results["Name"].unique()
unique_names = ['gili', 'liat', 'ariel', 'asaf']

# get all tasks with multiple annotations
# remove asaf annotations
# all_results = all_results[all_results["Name"] != "ariel"]
multiple_annotations = all_results["Task"].value_counts()[all_results["Task"].value_counts() > 1].index
df_multiple_annotations = all_results[all_results["Task"].isin(multiple_annotations)]
grouped = df_multiple_annotations.groupby("Task")

correctness = np.full((len(df_multiple_annotations), len(unique_names)), np.nan)
completeness = np.full((len(df_multiple_annotations), len(unique_names)), np.nan)
independence = np.full((len(df_multiple_annotations), len(unique_names)), np.nan)

names_to_index = {name: i for i, name in enumerate(unique_names)}
i = 0
for task, group in grouped:
    print(group[["Name", "Correctness", "Completeness", "Independence"]])
    # shuffle rows to avoid bias
    for _, row in group.iterrows():
        annotator_index = names_to_index[row["Name"]]
        correctness[i, annotator_index] = row["Correctness"]
        completeness[i, annotator_index] = row["Completeness"]
        independence[i, annotator_index] = row["Independence"]
    i += 1

alpha = krippendorff.alpha(reliability_data=correctness, level_of_measurement='ordinal')
print(f"Krippendorff's Alpha correctness: {alpha:.3f}")
alpha = krippendorff.alpha(reliability_data=completeness, level_of_measurement='ordinal')
print(f"Krippendorff's Alpha completeness: {alpha:.3f}")
alpha = krippendorff.alpha(reliability_data=independence, level_of_measurement='ordinal')
print(f"Krippendorff's Alpha independence: {alpha:.3f}")

correctness_mean = []
for i in range(len(unique_names)):
    not_nan = ~np.isnan(correctness[:,i])
    for j in range(i+1, len(unique_names)):
        not_nan_j = ~np.isnan(correctness[:,j])
        not_nan_both = not_nan & not_nan_j
        r1 = correctness[not_nan_both,i] + completeness[not_nan_both,i] + independence[not_nan_both,i]
        r2 = correctness[not_nan_both,j] + completeness[not_nan_both,j] + independence[not_nan_both,j]
        # calculate accuracy
        accuracy = np.mean(r1 == r2)
        mean_diff = np.mean(np.abs(r1 - r2))
        # print(f"Accuracy overall {unique_names[i]}, {unique_names[j]}: {accuracy:.3f}")
        print(f"Mean difference overall {unique_names[i]}, {unique_names[j]}: {mean_diff:.3f}")
#         weighted_kappa = cohen_kappa_score(r1, r2, weights='quadratic')
#         print(f"Weighted Cohen's Kappa (Linear) {unique_names[i]}, {unique_names[j]}: {weighted_kappa:.3f}")
#         correctness_mean.append(weighted_kappa)
# print("Mean Weighted Cohen's Kappa (Linear):", np.mean(correctness_mean))


df_llm_aaj_annotations = pd.read_csv("results/llm_aaj_annotations_asaf.csv")
y_true = df_llm_aaj_annotations["binary_score"].tolist()
y_pred = df_llm_aaj_annotations["human_eval"].tolist()
hamming = hamming_loss(y_true, y_pred)
print(f"Hamming Loss: {hamming:.4f}")


accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# cohen's kappa
kappa = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's Kappa: {kappa:.4f}")