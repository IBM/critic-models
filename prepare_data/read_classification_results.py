import json
import pandas as pd

with open("/Users/arielgera/Documents/Workspaces/critic-models/output/constraint-classification-via-rits-llama3.3-70b.wild-if-eval.json", "r") as f:
    j = json.load(f)

with open("/prepare_data/categories.json") as f:
    cats = json.load(f)

df = pd.DataFrame([{"constraint": c, "categories": c_str} for c, c_str in j.items()])
print(len(df))
df = df[df["categories"].apply(lambda x: len(x)<20 or "Other" in x)]
print(len(df))
df['categories'] = df['categories'].apply(lambda x: [y.replace("Category:", "").strip() for y in x.split(",") if len(y.strip())>0])
df['categories'] = df['categories'].apply(lambda l: [cats[int(x)]["name"] if "Other" not in x else x for x in l])
for cat_dict in cats:
    cat_name = cat_dict["name"]
    df[cat_name] = df['categories'].apply(lambda l: int(cat_name in l))
df["Other"] = df['categories'].apply(lambda l: int(len(l)==1 and "Other" in l[0]))

df.to_csv("categories.csv", index=False)