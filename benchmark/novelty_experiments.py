"""Six extra analyses for the paper (all open-source, seed 42, no test-set decisions).

  inflation    : how far split luck and common shortcuts inflate test accuracy (100 random 80/20 splits)
  conformal    : split-conformal crop shortlists inside 5x10 repeated CV (coverage vs set size)
  subsets      : which measurements are needed (feature subsets, 5x10 repeated CV)
  noise        : accuracy under multiplicative input noise (locked split, 10 draws per level)
  significance : per-fold scores + Nadeau-Bengio corrected repeated-CV t-test
  synthetic    : does the dataset look independently generated per class?
  pareto_acc   : RF accuracy vs number of trees (5x10 repeated CV)
  pareto_speed : RF latency vs trees, scikit-learn vs ONNX Runtime, 1 thread (run on an idle machine)
Each stage writes paper/assets/novelty_<stage>.json
"""
import os

for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from accuracy_push import (cat, et, knn, lgbm, load, mlp, pipe, rf, svm, xgb)  # noqa: E402
from sklearn.ensemble import StackingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
A = BASE / "paper" / "assets"
SEED = 42
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def save(stage, data):
    (A / f"novelty_{stage}.json").write_text(json.dumps(data, indent=2, default=float))
    print(stage, "saved", flush=True)


def flat(p):
    return np.asarray(p).ravel()


def folds(X, y):
    return list(RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=SEED).split(X, y))


def stacking():
    return pipe(StackingClassifier(
        estimators=[("rf", rf(n=500)), ("et", et(n=700)), ("svc", svm()), ("mlp", mlp())],
        final_estimator=LogisticRegression(max_iter=2000), cv=3, stack_method="predict_proba", n_jobs=1))


# ------------------------------------------------------------------ 1. inflation
def stage_inflation():
    X, y = load()
    pool = {  # a typical "try many models" pool
        "RF-100": lambda: pipe(rf(n=100)), "RF-300": lambda: pipe(rf(n=300)), "RF-700": lambda: pipe(rf()),
        "ET-300": lambda: pipe(et(n=300)), "ET-1200": lambda: pipe(et()),
        "XGB": lambda: pipe(xgb()), "XGB-deep": lambda: pipe(xgb(depth=8, n=500)),
        "LGBM": lambda: pipe(lgbm()), "LGBM-big": lambda: pipe(lgbm(n=600, leaves=31)),
        "SVM-C10": lambda: pipe(svm(C=10)), "SVM-C50": lambda: pipe(svm(C=50)),
        "kNN-1": lambda: pipe(knn(k=1)), "kNN-5": lambda: pipe(knn(k=5)),
    }
    seeds = list(range(100))
    honest, best_of_pool, best_name = [], [], []
    knn_proper, knn_leak, svm_proper, svm_leak = [], [], [], []
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    for s in seeds:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=s, stratify=y)
        accs = {}
        for name, make in pool.items():
            accs[name] = float((flat(make().fit(X[tr], y[tr]).predict(X[te])) == y[te]).mean())
        honest.append(accs["RF-700"])
        b = max(accs, key=accs.get); best_of_pool.append(accs[b]); best_name.append(b)
        # leakage: scalers fitted on ALL rows (train+test) before splitting
        Xl = StandardScaler().fit_transform(MinMaxScaler().fit_transform(X))
        knn_proper.append(accs["kNN-5"])                        # scalers fitted on training rows only
        knn_leak.append(float((flat(knn(k=5).fit(Xl[tr], y[tr]).predict(Xl[te])) == y[te]).mean()))
        svm_proper.append(accs["SVM-C10"])
        svm_leak.append(float((flat(svm(C=10).fit(Xl[tr], y[tr]).predict(Xl[te])) == y[te]).mean()))
        if s % 10 == 9:
            print("seed", s, np.mean(honest), np.mean(best_of_pool), flush=True)
    h, b = np.array(honest), np.array(best_of_pool)
    out = {
        "n_splits": len(seeds), "test_n": 440,
        "honest_rf": {"accs": h.tolist(), "mean": h.mean(), "std": h.std(), "min": h.min(), "max": h.max(),
                      "share_ge_9977": float((h >= 0.99772).mean()), "share_100": float((h == 1).mean())},
        "test_selected_pool": {"accs": b.tolist(), "mean": b.mean(), "max": b.max(),
                               "share_ge_9977": float((b >= 0.99772).mean()), "share_100": float((b == 1).mean()),
                               "pool": list(pool), "winners": {n: best_name.count(n) for n in set(best_name)}},
        "seed_shopping": {"rf_max_over_seeds": h.max(), "pool_max_over_seeds": b.max()},
        "scaling_leak": {"knn_proper": np.mean(knn_proper), "knn_leak": np.mean(knn_leak),
                         "svm_proper": np.mean(svm_proper), "svm_leak": np.mean(svm_leak),
                         "trees_note": "tree models are invariant to monotone per-feature scaling"},
    }
    save("inflation", out)


# ------------------------------------------------------------------ 2. conformal
def stage_conformal(alphas=(0.05, 0.02, 0.01, 0.005, 0.002)):
    """Split-conformal adaptive sets: score = probability mass of the classes ranked strictly above the true
    class (APS with U=0). A class enters the set if the mass ranked above it is <= q, so the top-ranked crop is
    always included and extra crops are added only when the model is unsure. Calibration = 5-fold OOF on
    each outer-train fold."""
    X, y = load()
    classes = np.unique(y)
    res = {a: {"cov": [], "size": [], "single": []} for a in alphas}
    top1, top2 = [], []
    from collections import Counter
    multi = Counter()

    def aps_scores(P, yi):
        order = np.argsort(-P, axis=1); cum = np.cumsum(np.take_along_axis(P, order, 1), 1)
        rank = np.argmax(order == yi[:, None], axis=1)
        return cum[np.arange(len(yi)), rank] - P[np.arange(len(yi)), yi]

    for k, (tr, te) in enumerate(folds(X, y)):
        oof = np.zeros((len(tr), len(classes)))
        for itr, ical in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X[tr], y[tr]):
            oof[ical] = pipe(rf()).fit(X[tr][itr], y[tr][itr]).predict_proba(X[tr][ical])
        cal = aps_scores(oof, np.searchsorted(classes, y[tr]))
        P = pipe(rf()).fit(X[tr], y[tr]).predict_proba(X[te]); yt = np.searchsorted(classes, y[te])
        order = np.argsort(-P, axis=1); cum = np.cumsum(np.take_along_axis(P, order, 1), 1)
        top1.append(float((order[:, 0] == yt).mean())); top2.append(float((order[:, :2] == yt[:, None]).any(1).mean()))
        n = len(cal)
        for a in alphas:
            q = np.quantile(cal, min(1.0, np.ceil((n + 1) * (1 - a)) / n), method="higher")
            before = cum - np.take_along_axis(P, order, 1)            # mass ranked strictly above each class
            k_incl = np.maximum((before <= q + 1e-12).sum(1), 1)
            cover = np.array([yt[i] in order[i, :k_incl[i]] for i in range(len(yt))])
            res[a]["cov"].append(float(cover.mean())); res[a]["size"].append(float(k_incl.mean()))
            res[a]["single"].append(float((k_incl == 1).mean()))
            if a == 0.002:
                for i in np.where(k_incl > 1)[0]:
                    multi[tuple(sorted(classes[order[i, :k_incl[i]]].tolist()))] += 1
        if k % 10 == 9:
            print("fold", k + 1, {a: (round(np.mean(res[a]["cov"]), 4), round(np.mean(res[a]["size"]), 3)) for a in alphas}, flush=True)
    out = {"method": "split-conformal adaptive sets (APS, U=0), RF-700, calibration = 5-fold OOF per outer-train fold, 5x10 CV",
           "top1_accuracy": float(np.mean(top1)), "top2_accuracy": float(np.mean(top2)),
           "by_alpha": {str(a): {k: float(np.mean(v)) for k, v in res[a].items()} for a in alphas},
           "multi_sets_alpha_0.002": [[list(k), v] for k, v in multi.most_common(8)],
           "multi_set_count_alpha_0.002": int(sum(multi.values()))}
    save("conformal", out)


# ------------------------------------------------------------------ 3. feature subsets
SUBSETS = {
    "Climate (T, H, R)": ["temperature", "humidity", "rainfall"],
    "Climate + pH": ["temperature", "humidity", "rainfall", "ph"],
    "Soil (N, P, K, pH)": ["N", "P", "K", "ph"],
    "Climate + N, P, K": ["N", "P", "K", "temperature", "humidity", "rainfall"],
    "All except rainfall": ["N", "P", "K", "temperature", "humidity", "ph"],
    "All seven": FEATURES,
}


def stage_subsets():
    X, y = load(); F = folds(X, y); out = {}
    for name, cols in SUBSETS.items():
        idx = [FEATURES.index(c) for c in cols]; accs = []; wrong = {}
        for tr, te in F:
            p = flat(pipe(rf()).fit(X[tr][:, idx], y[tr]).predict(X[te][:, idx]))
            accs.append(float((p == y[te]).mean()))
            for t, q in zip(y[te][p != y[te]], p[p != y[te]]):
                wrong[f"{t}->{q}"] = wrong.get(f"{t}->{q}", 0) + 1
        out[name] = {"features": cols, "mean": float(np.mean(accs)), "std": float(np.std(accs)),
                     "top_confusions": sorted(wrong.items(), key=lambda kv: -kv[1])[:5]}
        print(name, round(out[name]["mean"] * 100, 2), flush=True)
    save("subsets", out)


# ------------------------------------------------------------------ 4. noise robustness
def stage_noise(levels=(0.0, 0.02, 0.05, 0.10, 0.20), draws=10):
    X, y = load()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)
    models = {"Random Forest": lambda: pipe(rf()), "Extra Trees": lambda: pipe(et()), "LightGBM": lambda: pipe(lgbm()),
              "XGBoost": lambda: pipe(xgb()), "CatBoost": lambda: pipe(cat()), "SVM (RBF)": lambda: pipe(svm()),
              "k-NN": lambda: pipe(knn()), "MLP": lambda: pipe(mlp()), "Stacking": stacking}
    out = {"noise": "x * (1 + sigma * N(0,1)) on every test feature; training data clean", "levels": list(levels), "models": {}}
    for name, make in models.items():
        m = make().fit(X[tr], y[tr]); curve = []
        for s in levels:
            rng = np.random.default_rng(SEED); accs = []
            for _ in range(draws if s > 0 else 1):
                Xn = (X[te] * (1 + s * rng.standard_normal(X[te].shape))).astype(np.float32)
                accs.append(float((flat(m.predict(Xn)) == y[te]).mean()))
            curve.append(float(np.mean(accs)))
        out["models"][name] = curve
        print(name, [round(c * 100, 1) for c in curve], flush=True)
    save("noise", out)


# ------------------------------------------------------------------ 5. significance
def stage_significance():
    X, y = load(); F = folds(X, y)
    models = {"Random Forest": lambda: pipe(rf()), "Stacking": stacking, "Extra Trees": lambda: pipe(et()),
              "CatBoost": lambda: pipe(cat()), "XGBoost": lambda: pipe(xgb()), "LightGBM": lambda: pipe(lgbm())}
    per = {}
    for name, make in models.items():
        per[name] = [float((flat(make().fit(X[tr], y[tr]).predict(X[te])) == y[te]).mean()) for tr, te in F]
        print(name, round(np.mean(per[name]) * 100, 3), flush=True)
    J, ratio = len(F), 1 / 9  # n_test / n_train for 10-fold

    def nb(a, b):
        d = np.array(per[a]) - np.array(per[b]); v = d.var(ddof=1)
        if v == 0:
            return 0.0, 1.0
        t = d.mean() / np.sqrt((1 / J + ratio) * v); p = 2 * stats.t.sf(abs(t), J - 1)
        return float(t), float(p)
    tests = {f"{a} vs {b}": dict(zip(["t", "p"], nb(a, b))) for a in models for b in models if a < b}
    save("significance", {"per_fold": per, "means": {k: float(np.mean(v)) for k, v in per.items()},
                          "test": "Nadeau-Bengio corrected resampled t-test, J=50, n2/n1=1/9", "pairs": tests})


# ------------------------------------------------------------------ 6. synthetic-data check
def stage_synthetic():
    X, y = load(); classes = np.unique(y)
    within, pvals = [], []
    unif_ok = norm_ok = total = 0
    for c in classes:
        Xc = X[y == c]
        r = np.corrcoef(Xc.T)
        for i in range(7):
            for j in range(i + 1, 7):
                within.append(abs(r[i, j])); pvals.append(stats.pearsonr(Xc[:, i], Xc[:, j])[1])
        for f in range(7):
            v = Xc[:, f]; total += 1
            if v.max() > v.min():
                unif_ok += stats.kstest((v - v.min()) / (v.max() - v.min()), "uniform").pvalue > 0.05
            norm_ok += stats.shapiro(v).pvalue > 0.05
    g = np.corrcoef(X.T); glob = [abs(g[i, j]) for i in range(7) for j in range(i + 1, 7)]
    pvals = np.array(pvals)
    out = {"within_class_mean_abs_r": float(np.mean(within)), "within_class_max_abs_r": float(np.max(within)),
           "share_pairs_significant_p05": float((pvals < 0.05).mean()), "expected_if_independent": 0.05,
           "n_pairs": int(len(pvals)), "global_mean_abs_r": float(np.mean(glob)),
           "global_PK_r": float(g[1, 2]),
           "per_class_feature_uniform_not_rejected": int(unif_ok), "per_class_feature_normal_not_rejected": int(norm_ok),
           "per_class_feature_total": int(total),
           "exact_duplicate_rows": int(len(X) - len(np.unique(X, axis=0))),
           "integer_valued": {f: bool(np.all(np.mod(X[:, i], 1) == 0)) for i, f in enumerate(FEATURES)}}
    save("synthetic", out); print(json.dumps(out, indent=1))


# ------------------------------------------------------------------ 7. accuracy/speed Pareto
TREES = [10, 25, 50, 100, 200, 400, 700]


def stage_pareto_acc():
    X, y = load(); F = folds(X, y); out = {}
    for n in TREES:
        accs = [float((flat(pipe(rf(n=n)).fit(X[tr], y[tr]).predict(X[te])) == y[te]).mean()) for tr, te in F]
        out[str(n)] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
        print(n, round(out[str(n)]["mean"] * 100, 3), flush=True)
    save("pareto_acc", out)


def stage_pareto_speed(n_single=1000):
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    X, y = load()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)
    rows = [X[te][i:i + 1] for i in range(len(te))]
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    out = {}
    for n in TREES:
        p = Pipeline([("mm", MinMaxScaler()), ("sd", StandardScaler()),
                      ("clf", RandomForestClassifier(n_estimators=n, random_state=SEED, n_jobs=1))]).fit(X[tr], y[tr])
        onx = convert_sklearn(p, initial_types=[("float_input", FloatTensorType([None, 7]))], options={id(p): {"zipmap": False}})
        sess = ort.InferenceSession(onx.SerializeToString(), so, providers=["CPUExecutionProvider"])
        for i in range(30):
            p.predict(rows[i]); sess.run(["label"], {"float_input": rows[i]})
        t = []
        for i in range(n_single):
            t0 = time.perf_counter(); p.predict(rows[i % len(rows)]); t.append(time.perf_counter() - t0)
        u = []
        for i in range(n_single):
            t0 = time.perf_counter(); sess.run(["label"], {"float_input": rows[i % len(rows)]}); u.append(time.perf_counter() - t0)
        agree = bool(np.all(flat(sess.run(["label"], {"float_input": X[te]})[0]).astype(str) == flat(p.predict(X[te])).astype(str)))
        out[str(n)] = {"sklearn_ms": float(np.mean(t) * 1e3), "onnx_ms": float(np.mean(u) * 1e3),
                       "onnx_bytes": len(onx.SerializeToString()), "agree_440": agree}
        print(n, out[str(n)], flush=True)
    save("pareto_speed", out)


if __name__ == "__main__":
    for st in sys.argv[1:]:
        globals()[f"stage_{st}"]()
