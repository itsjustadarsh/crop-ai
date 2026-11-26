"""Accuracy push without leakage.

Stage `locked`  : same protocol as the paper. Locked stratified 80/20 split (seed 42); every
                  hyperparameter / feature-set / ensemble-composition decision is made by inner
                  3-fold CV on the 1,760 training rows; the 440 test rows are scored once at the end.
Stage `repeated`: 5 x stratified 10-fold CV over all 2,200 rows with FIXED configurations that
                  were defined before evaluation (library defaults or settings stated below), so no
                  outer test fold ever influences a choice. Also records per-sample error counts.
Stage `ceiling` : nearest-neighbour analysis of samples that (almost) every model gets wrong.

Only open-source, pip-installable libraries are used (no accounts, keys or gated weights).
Outputs: paper/assets/accuracy_push.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx_winner_benchmark import FEATURES, load_dataset  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "paper" / "assets" / os.environ.get("PUSH_OUT", "accuracy_push.json")
SEED = 42
N_TRIALS = 30


# ------------------------------------------------------------------ building blocks
def engineer(X):
    """N:P:K ratios and climate interactions (raw-unit features)."""
    N, P, K, T, H, pH, R = [X[:, i] for i in range(7)]
    extra = np.column_stack([N / (P + 1), N / (K + 1), P / (K + 1), T * H / 100, H * R / 100, np.log1p(R)])
    return np.hstack([X, extra]).astype(np.float32)


def pipe(clf, fe=False):
    steps = [("fe", FunctionTransformer(engineer))] if fe else []
    return Pipeline(steps + [("mm", MinMaxScaler()), ("sd", StandardScaler()), ("clf", clf)])


class LabelEncoded(BaseEstimator, ClassifierMixin):
    """Wrap XGBoost so it accepts string labels."""
    def __init__(self, est=None):
        self.est = est

    def fit(self, X, y):
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.est_ = clone(self.est).fit(X, yi)
        return self

    def predict_proba(self, X):
        return self.est_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.asarray(self.est_.predict(X)).ravel().astype(int)]


def rf(**k): return RandomForestClassifier(n_estimators=k.get("n", 700), random_state=SEED, n_jobs=-1)
def et(**k): return ExtraTreesClassifier(n_estimators=k.get("n", 1200), random_state=SEED, n_jobs=-1)
def svm(**k): return SVC(kernel="rbf", C=k.get("C", 20.0), probability=True, random_state=SEED)
def mlp(**k): return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=SEED)
def knn(**k): return KNeighborsClassifier(n_neighbors=k.get("k", 5), weights=k.get("w", "distance"))
def lgbm(**k):
    return LGBMClassifier(n_estimators=k.get("n", 300), learning_rate=k.get("lr", 0.05), num_leaves=k.get("leaves", 15),
                          min_child_samples=k.get("mcs", 10), subsample=k.get("sub", 1.0), subsample_freq=1,
                          colsample_bytree=k.get("col", 1.0), reg_lambda=k.get("l2", 0.0),
                          random_state=SEED, n_jobs=-1, verbose=-1)
def cat(**k):
    return LabelEncoded(CatBoostClassifier(iterations=k.get("n", 500), learning_rate=k.get("lr", 0.08), depth=k.get("depth", 6),
                                           l2_leaf_reg=k.get("l2", 3.0), random_seed=SEED, verbose=0, thread_count=-1))
def xgb(**k):
    return LabelEncoded(XGBClassifier(n_estimators=k.get("n", 300), max_depth=k.get("depth", 5),
                                      learning_rate=k.get("lr", 0.05), subsample=k.get("sub", 1.0),
                                      colsample_bytree=k.get("col", 1.0), random_state=SEED,
                                      eval_metric="mlogloss", verbosity=0, n_jobs=-1))


SPACES = {
    "LightGBM": (lgbm, lambda t: {"n": t.suggest_int("n", 100, 600, step=100), "lr": t.suggest_float("lr", 0.02, 0.2, log=True),
                                  "leaves": t.suggest_int("leaves", 7, 63), "mcs": t.suggest_int("mcs", 3, 30),
                                  "sub": t.suggest_float("sub", 0.6, 1.0), "col": t.suggest_float("col", 0.6, 1.0),
                                  "l2": t.suggest_float("l2", 1e-3, 10, log=True)}),
    "CatBoost": (cat, lambda t: {"n": t.suggest_int("n", 200, 800, step=200), "lr": t.suggest_float("lr", 0.03, 0.3, log=True),
                                 "depth": t.suggest_int("depth", 4, 8), "l2": t.suggest_float("l2", 1, 10, log=True)}),
    "XGBoost": (xgb, lambda t: {"n": t.suggest_int("n", 100, 500, step=100), "depth": t.suggest_int("depth", 3, 8),
                                "lr": t.suggest_float("lr", 0.02, 0.3, log=True), "sub": t.suggest_float("sub", 0.6, 1.0),
                                "col": t.suggest_float("col", 0.6, 1.0)}),
    "k-NN": (knn, lambda t: {"k": t.suggest_int("k", 1, 15), "w": t.suggest_categorical("w", ["uniform", "distance"])}),
}
TRIALS = {"LightGBM": N_TRIALS, "CatBoost": 15, "XGBoost": N_TRIALS, "k-NN": 30}


def inner_cv(make, X, y, n_splits=3):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    accs = [float((np.asarray(make().fit(X[tr], y[tr]).predict(X[va])).ravel() == y[va]).mean()) for tr, va in cv.split(X, y)]
    return float(np.mean(accs)), float(np.std(accs))


def load():
    df = load_dataset()
    return df[FEATURES].to_numpy(np.float32), df["label"].to_numpy()


def save(key, value):
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[key] = value
    OUT.write_text(json.dumps(data, indent=2, default=float))


# ------------------------------------------------------------------ stage A: locked split
def stage_locked():
    X, y = load()
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    assert not set(tr_idx) & set(te_idx)
    Xtr, ytr, Xte, yte = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]   # tuning below only sees Xtr/ytr
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    res, best_cfg = {}, {}

    def record(name, make, cv, extra=None):
        t0 = time.perf_counter(); model = make().fit(Xtr, ytr); fit_s = time.perf_counter() - t0
        pred = np.asarray(model.predict(Xte)).ravel()
        r = {"cv_mean": cv[0], "cv_std": cv[1], "test_acc": float((pred == yte).mean()),
             "test_correct": int((pred == yte).sum()), "fit_seconds": fit_s,
             "test_errors": [[str(a), str(b)] for a, b in zip(yte[pred != yte], pred[pred != yte])]}
        r.update(extra or {})
        res[name] = r; save("locked", res)
        print(name, json.dumps({k: r[k] for k in ["cv_mean", "test_acc", "test_correct"]}), flush=True)

    # 1. tuned single models (raw features), objective = inner CV on Xtr only
    for name, (factory, space) in SPACES.items():
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(lambda t: inner_cv(lambda: pipe(factory(**space(t))), Xtr, ytr)[0], n_trials=TRIALS[name])
        p = study.best_params; best_cfg[name] = (factory, p)
        record(name + " (tuned)", lambda: pipe(factory(**p)), inner_cv(lambda: pipe(factory(**p)), Xtr, ytr),
               {"params": p})
    for name, factory in [("Random Forest", rf), ("Extra Trees", et), ("SVM (RBF)", svm)]:
        best_cfg[name] = (factory, {})

    # 2. feature engineering: kept only if inner CV improves
    fe_scores = {}
    for name in ["Random Forest", "Extra Trees", "LightGBM", "XGBoost"]:
        f, p = best_cfg[name]
        base = inner_cv(lambda: pipe(f(**p)), Xtr, ytr)[0]
        eng = inner_cv(lambda: pipe(f(**p), fe=True), Xtr, ytr)[0]
        fe_scores[name] = {"raw_cv": base, "engineered_cv": eng}
    use_fe = np.mean([v["engineered_cv"] - v["raw_cv"] for v in fe_scores.values()]) > 0
    save("feature_engineering", {"inner_cv": fe_scores, "adopted": bool(use_fe)})
    print("feature engineering adopted:", use_fe, flush=True)

    # 3. super-stack: base learners = top-5 by inner CV (from step 1 + fixed models), LR meta
    cv_rank = {}
    for name, (f, p) in best_cfg.items():
        cv_rank[name] = res.get(name + " (tuned)", res.get(name, {})).get("cv_mean") or inner_cv(lambda: pipe(f(**p)), Xtr, ytr)[0]
    top = sorted(cv_rank, key=cv_rank.get, reverse=True)[:5]
    save("superstack_selection", {"inner_cv_by_model": cv_rank, "chosen": top})

    def superstack():
        ests = [(n.replace(" ", "_"), best_cfg[n][0](**best_cfg[n][1])) for n in top]
        return pipe(StackingClassifier(estimators=ests, final_estimator=LogisticRegression(C=10, max_iter=3000),
                                       cv=5, stack_method="predict_proba", n_jobs=1), fe=use_fe)
    record("Super-stack", superstack, inner_cv(superstack, Xtr, ytr), {"base_learners": top, "feature_engineering": bool(use_fe)})


# ------------------------------------------------------------------ stage B: repeated CV
def fixed_models():
    m = {
        "Random Forest": lambda: pipe(rf()),
        "Extra Trees": lambda: pipe(et()),
        "LightGBM": lambda: pipe(lgbm()),
        "CatBoost": lambda: pipe(cat()),
        "XGBoost": lambda: pipe(xgb()),
        "k-NN (k=5)": lambda: pipe(knn()),
        "Stacking": lambda: pipe(StackingClassifier(
            estimators=[("rf", rf(n=500)), ("et", et(n=700)), ("svc", svm()), ("mlp", mlp())],
            final_estimator=LogisticRegression(max_iter=2000), cv=3, stack_method="predict_proba", n_jobs=1)),
        "Super-stack (fixed)": lambda: pipe(StackingClassifier(
            estimators=[("rf", rf()), ("et", et()), ("lgbm", lgbm()), ("cat", cat()), ("svc", svm())],
            final_estimator=LogisticRegression(C=10, max_iter=3000), cv=5, stack_method="predict_proba", n_jobs=1)),
    }
    return m


def stage_repeated():
    X, y = load()
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=SEED)
    folds = list(rskf.split(X, y))
    out = json.loads(OUT.read_text()).get("repeated", {}) if OUT.exists() else {}
    for name, make in fixed_models().items():
        if name in out:
            continue
        t0 = time.perf_counter(); accs = []; wrong = np.zeros(len(y), dtype=int)
        for tr, te in folds:
            assert not set(tr) & set(te)
            pred = np.asarray(make().fit(X[tr], y[tr]).predict(X[te])).ravel()
            accs.append(float((pred == y[te]).mean())); wrong[te[pred != y[te]]] += 1
        out[name] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "folds": len(accs),
                     "errors_total": int(wrong.sum()), "predictions_total": len(folds) * len(folds[0][1]),
                     "per_sample_errors": {int(i): int(c) for i, c in enumerate(wrong) if c},
                     "seconds": time.perf_counter() - t0}
        save("repeated", out)
        print(name, f"{out[name]['mean']*100:.3f} ± {out[name]['std']*100:.3f}", out[name]["errors_total"], flush=True)


# ------------------------------------------------------------------ stage C: ceiling analysis
def stage_ceiling():
    X, y = load()
    rep = json.loads(OUT.read_text())["repeated"]
    models = list(rep)
    n_rep = 5
    counts = {}
    for m in models:
        for i, c in rep[m]["per_sample_errors"].items():
            counts.setdefault(int(i), {})[m] = c
    hard = sorted(counts, key=lambda i: -sum(counts[i].values()))
    Xs = StandardScaler().fit_transform(MinMaxScaler().fit_transform(X))
    nn = NearestNeighbors(n_neighbors=6).fit(Xs)
    rows = []
    for i in hard:
        frac_models = sum(1 for m in models if counts[i].get(m, 0) >= 1) / len(models)
        always = sum(1 for m in models if counts[i].get(m, 0) == n_rep)
        _, nb = nn.kneighbors(Xs[i:i + 1]); nb = [j for j in nb[0] if j != i][:5]
        rows.append({"index": i, "true": str(y[i]), "features": dict(zip(FEATURES, X[i].round(2).tolist())),
                     "models_erring_fraction": frac_models, "models_always_wrong": always,
                     "neighbour_labels": [str(y[j]) for j in nb]})
    always_all = [r for r in rows if r["models_always_wrong"] == len(models)]
    always_most = [r for r in rows if r["models_always_wrong"] >= len(models) / 2]
    save("ceiling", {"n_models": len(models), "samples_with_any_error": len(rows),
                     "samples_wrong_for_every_model_every_repeat": len(always_all),
                     "samples_wrong_for_majority_every_repeat": len(always_most),
                     "implied_ceiling_accuracy": 1 - len(always_all) / len(y), "hardest": rows[:15]})
    print(json.dumps({"any_error": len(rows), "always_all": len(always_all), "always_most": len(always_most)}), flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "locked"
    {"locked": stage_locked, "repeated": stage_repeated, "ceiling": stage_ceiling}[stage]()
