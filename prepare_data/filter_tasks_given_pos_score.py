import os.path
from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
import numpy as np
from utils.filter_data_consts import get_bad_words


def run(path_to_scores_file, out_dir, percentile, threshold):
    with open(path_to_scores_file, 'rt') as f:
        scores = json.load(f)

    scores_filtered = {}
    bad_words = get_bad_words()
    for sample in scores:
        skip = False
        for word in bad_words:
            if word in sample:
                skip = True
                break
        if skip:
            continue
        scores_filtered[sample] = scores[sample]
    print(f"{path_to_scores_file}: filtered {len(scores)-len(scores_filtered)} samples which were offensive ({round(len(scores_filtered)/len(scores)*100, 2)}%)")

    if percentile > 0:
        # filter by percentile

        pos_scores = np.array([scores_filtered[sample]["pos_score"] for sample in scores_filtered
                               if scores_filtered[sample]["pos_score"] != "ERR"])
        threshold = np.percentile(pos_scores, percentile)
        out_path = os.path.join(out_dir, f"filtered_{round(percentile/100, 2)}percentile_{round(threshold,2)}threshold.json")
        print(f"{path_to_scores_file} {percentile} top pos scores requires a threshold of {threshold}")
    else:
        out_path = os.path.join(out_dir, f"filtered_{threshold}threshold.json")
    print(f"data will be saved in {out_path}")

    positive_filtered = {task: scores_filtered[task] for task in scores_filtered
                         if scores_filtered[task]["pos_score"] != 'ERR' and scores_filtered[task]["pos_score"]  > threshold}

    with open(out_path, 'wt') as f:
        str_to_dump = json.dumps(positive_filtered, indent=2)
        f.write(str_to_dump)
    print(f"{len(positive_filtered)} marked positive from {path_to_scores_file}")

    pos_scores = [np.array([scores_filtered[res]["pos_score"]
                            for res in scores_filtered if scores_filtered[res]["pos_score"] != 'ERR'],
                           dtype=np.float32)
                  ]
    plt.violinplot(pos_scores, showmeans=True)
    plt.xticks([1], labels=[path_to_scores_file])
    path_to_vis = path_to_scores_file.replace(".json", "_pos_score_dist.png")
    plt.savefig(path_to_vis)
    print(f"distribution plot in {path_to_vis}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--percentile", type=float, default=-1.0)
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--out_dir")
    parser.add_argument("--scores")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.threshold == -1 and args.percentile == -1:
        raise RuntimeError("at least one of threshold or percentile should be set")
    run(args.scores, args.out_dir, args.percentile, args.threshold)
