import numpy as np
import json

import pandas as pd
from nomic import atlas
from argparse import ArgumentParser
from nomic import embed


def main(embeddings_file, texts_file):
    # Load embeddings
    embeddings = np.load(embeddings_file)

    # Load texts
    with open(texts_file, "r", encoding="utf-8") as f:
        texts = json.load(f)

    # Upload to Nomic Atlas
    atlas.map_data(
        data=pd.DataFrame([{"text": text} for text in texts[:1000]]),
        indexed_field="text",
        identifier="Embedding Visualization v3",
        description="Visualization of embeddings loaded from .npy and .json files."
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--embeddings", type=str, help="Path to the .npy file containing embeddings")
    parser.add_argument("--list", type=str, help="Path to the .json file containing corresponding texts")
    args = parser.parse_args()
    main(args.embeddings, args.list)
