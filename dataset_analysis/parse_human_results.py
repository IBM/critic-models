import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np
import krippendorff


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

# get all tasks with multiple annotations
multiple_annotations = all_results["Task"].value_counts()[all_results["Task"].value_counts() > 1].index
df_multiple_annotations = all_results[all_results["Task"].isin(multiple_annotations)]
grouped = df_multiple_annotations.groupby("Task")

annotator_1 = []
annotator_2 = []

for task, group in grouped:
    print(group["Name"])
    # shuffle rows to avoid bias
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            annotator_1.extend(group.iloc[i][["Correctness", "Completeness", "Independence"]].values)
            annotator_2.extend(group.iloc[j][["Correctness", "Completeness", "Independence"]].values)

ratings = np.array([annotator_1, annotator_2])
alpha = krippendorff.alpha(reliability_data=ratings, level_of_measurement='ordinal')
print(f"Krippendorff's Alpha: {alpha:.3f}")

