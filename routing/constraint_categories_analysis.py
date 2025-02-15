import numpy as np
import pandas as pd
CSV_PATH = "/Users/gililior/research/datasets/arena_data_v2/categories.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    labels = list(df.columns[2:])
    categories_mapping = {}
    for category in labels:
        index_for_label = np.where(df[category] == 1)[0]
        categories_mapping[category] = index_for_label





if __name__ == '__main__':
    main()