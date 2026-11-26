"""Accuracy-ceiling analysis from the 5x10 repeated-CV error counts.

For every sample: how many models misclassify it, and in how many of the 5 repeats.
A sample that every model gets wrong in every repeat is effectively unlearnable from these
seven features. Its nearest neighbours (scaled feature space) show which class it resembles.
Input : paper/assets/accuracy_push_repeated.json   Output: paper/assets/ceiling.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx_winner_benchmark import FEATURES, load_dataset  # noqa: E402

A = Path(__file__).resolve().parent.parent / "paper" / "assets"
REPEATS = 5


def main():
    rep = json.loads((A / "accuracy_push_repeated.json").read_text())["repeated"]
    models = list(rep)
    df = load_dataset()
    X = df[FEATURES].to_numpy(np.float32)
    y = df["label"].to_numpy()
    err = np.zeros((len(models), len(y)), dtype=int)
    for mi, m in enumerate(models):
        for i, c in rep[m]["per_sample_errors"].items():
            err[mi, int(i)] = c
    always_all = np.where((err == REPEATS).all(axis=0))[0]
    any_err = np.where(err.sum(axis=0) > 0)[0]
    majority = np.where((err == REPEATS).sum(axis=0) >= len(models) / 2)[0]
    Xs = StandardScaler().fit_transform(MinMaxScaler().fit_transform(X))
    nn = NearestNeighbors(n_neighbors=6).fit(Xs)
    hard = []
    for i in sorted(any_err, key=lambda i: -err[:, i].sum())[:10]:
        _, nb = nn.kneighbors(Xs[i:i + 1])
        nbl = [str(y[j]) for j in nb[0] if j != i][:5]
        hard.append({"index": int(i), "true": str(y[i]), "errors_by_model": dict(zip(models, err[:, i].tolist())),
                     "features": dict(zip(FEATURES, [round(float(v), 2) for v in X[i]])), "neighbour_labels": nbl,
                     "neighbours_same_class": sum(l == y[i] for l in nbl)})
    out = {"models": models, "n_samples": int(len(y)),
           "samples_with_any_error": int(len(any_err)),
           "samples_wrong_every_model_every_repeat": [int(i) for i in always_all],
           "samples_wrong_majority_every_repeat": [int(i) for i in majority],
           "oracle_ceiling_accuracy": float(1 - len(always_all) / len(y)),
           "best_single_model": max(models, key=lambda m: rep[m]["mean"]),
           "hardest": hard}
    (A / "ceiling.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ["samples_with_any_error", "samples_wrong_every_model_every_repeat",
                                          "samples_wrong_majority_every_repeat", "oracle_ceiling_accuracy"]}))
    for h in hard[:6]:
        print(h["index"], h["true"], h["features"], "nbrs:", h["neighbour_labels"], "err:", h["errors_by_model"])


if __name__ == "__main__":
    main()
