"""Leakage-free hyperparameter tuning for ensemble models.

Optuna objective = mean inner 3-fold stratified CV accuracy on the OUTER-TRAIN
split only. The outer test split (440 rows) is touched exactly once per
candidate, after tuning is finished. Scalers are fit inside every fold.

Outputs: paper/assets/tuned_ensembles.json
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx_winner_benchmark import (FEATURES, OUTER_SEED, fit_scalers,  # noqa: E402
                                   load_dataset, metric_dict, scale_pair)

BASE = Path(__file__).resolve().parent.parent
OUT_JSON = BASE / "paper" / "assets" / "tuned_ensembles.json"
OUT_PKL = BASE / "model" / "tuned_ensembles.pkl"
N_TRIALS = 20
SEED = OUTER_SEED


def stacking_lr(p):
    return StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=p["rf_n"], random_state=SEED, n_jobs=-1)),
            ("et", ExtraTreesClassifier(n_estimators=p["et_n"], random_state=SEED, n_jobs=-1)),
            ("svc", SVC(kernel="rbf", C=p["svc_C"], probability=True, random_state=SEED)),
            ("mlp", MLPClassifier(hidden_layer_sizes=p["mlp_h"], max_iter=400, random_state=SEED)),
        ],
        final_estimator=LogisticRegression(C=p["meta_C"], max_iter=2000, random_state=SEED),
        cv=3, stack_method="predict_proba", passthrough=p["passthrough"], n_jobs=-1,
    )


def stacking_rf(p):
    return StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=p["rf_n"], random_state=SEED, n_jobs=-1)),
            ("et", ExtraTreesClassifier(n_estimators=p["et_n"], random_state=SEED, n_jobs=-1)),
            ("xgb", XGBClassifier(n_estimators=p["xgb_n"], max_depth=p["xgb_depth"],
                                  learning_rate=p["xgb_lr"], random_state=SEED,
                                  eval_metric="mlogloss", verbosity=0, n_jobs=1)),
            ("svc", SVC(kernel="rbf", C=p["svc_C"], probability=True, random_state=SEED)),
        ],
        final_estimator=RandomForestClassifier(n_estimators=p["meta_n"], max_depth=p["meta_depth"],
                                               random_state=SEED, n_jobs=-1),
        cv=3, stack_method="predict_proba", n_jobs=-1,
    )


def soft_voting(p):
    return VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=p["rf_n"], random_state=SEED, n_jobs=-1)),
            ("et", ExtraTreesClassifier(n_estimators=p["et_n"], random_state=SEED, n_jobs=-1)),
            ("xgb", XGBClassifier(n_estimators=p["xgb_n"], max_depth=p["xgb_depth"],
                                  learning_rate=p["xgb_lr"], random_state=SEED,
                                  eval_metric="mlogloss", verbosity=0, n_jobs=1)),
            ("svc", SVC(kernel="rbf", C=p["svc_C"], probability=True, random_state=SEED)),
        ],
        voting="soft", weights=[p["w_rf"], p["w_et"], p["w_xgb"], p["w_svc"]], n_jobs=1,
    )


def space_stacking_lr(t):
    return {
        "rf_n": t.suggest_int("rf_n", 300, 900, step=200),
        "et_n": t.suggest_int("et_n", 500, 1300, step=200),
        "svc_C": t.suggest_float("svc_C", 5.0, 100.0, log=True),
        "mlp_h": t.suggest_categorical("mlp_h", [(128, 64), (256, 128)]),
        "meta_C": t.suggest_float("meta_C", 0.1, 100.0, log=True),
        "passthrough": t.suggest_categorical("passthrough", [False, True]),
    }


def space_trees_xgb_svc(t):
    return {
        "rf_n": t.suggest_int("rf_n", 300, 900, step=200),
        "et_n": t.suggest_int("et_n", 500, 1300, step=200),
        "xgb_n": t.suggest_int("xgb_n", 100, 400, step=100),
        "xgb_depth": t.suggest_int("xgb_depth", 3, 7),
        "xgb_lr": t.suggest_float("xgb_lr", 0.02, 0.3, log=True),
        "svc_C": t.suggest_float("svc_C", 5.0, 100.0, log=True),
    }


def space_stacking_rf(t):
    p = space_trees_xgb_svc(t)
    p["meta_n"] = t.suggest_int("meta_n", 100, 500, step=100)
    p["meta_depth"] = t.suggest_int("meta_depth", 4, 16)
    return p


def space_voting(t):
    p = space_trees_xgb_svc(t)
    for k in ["w_rf", "w_et", "w_xgb", "w_svc"]:
        p[k] = t.suggest_int(k, 1, 3)
    return p


CANDIDATES = {
    "Stacking (LR meta, tuned)": (stacking_lr, space_stacking_lr),
    "Stacking (RF meta, tuned)": (stacking_rf, space_stacking_rf),
    "Soft Voting (tuned)": (soft_voting, space_voting),
}


def inner_cv(factory, X, y):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    accs, f1s = [], []
    for tr, va in cv.split(X, y):
        mm, std = fit_scalers(X[tr])
        Xtr, Xva = scale_pair(mm, std, X[tr], X[va])
        m = factory().fit(Xtr, y[tr])
        r = metric_dict(y[va], m.predict(Xva))
        accs.append(r["accuracy"]); f1s.append(r["f1_macro"])
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))


def full_data_5fold(factory, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    s = []
    for tr, te in cv.split(X, y):
        mm, std = fit_scalers(X[tr])
        Xtr, Xte = scale_pair(mm, std, X[tr], X[te])
        s.append(float(accuracy_score(y[te], factory().fit(Xtr, y[tr]).predict(Xte))))
    return float(np.mean(s)), float(np.std(s))


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df = load_dataset()
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label_id"].to_numpy(dtype=np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    out, models = {"protocol": {
        "objective": "mean inner 3-fold stratified CV accuracy on outer-train (1,760 rows)",
        "sampler": f"TPE, seed={SEED}", "n_trials": N_TRIALS,
        "test_usage": "outer test (440 rows) evaluated once after tuning"}, "results": []}, {}

    for name, (build, space) in CANDIDATES.items():
        t0 = time.perf_counter()
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(lambda t: inner_cv(lambda: build(space(t)), Xtr, ytr)[0], n_trials=N_TRIALS)
        best = dict(study.best_params)
        if "mlp_h" in best:
            best["mlp_h"] = tuple(best["mlp_h"])
        factory = lambda: build(best)  # noqa: E731
        cv_mean, cv_std, cv_f1 = inner_cv(factory, Xtr, ytr)
        mm, std = fit_scalers(Xtr)
        Xtr_s, Xte_s = scale_pair(mm, std, Xtr, Xte)
        model = factory().fit(Xtr_s, ytr)
        test = metric_dict(yte, model.predict(Xte_s))
        r5_mean, r5_std = full_data_5fold(factory, X, y)
        row = {"model_name": name, "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()},
               "cv_mean_accuracy": cv_mean, "cv_std_accuracy": cv_std, "cv_mean_f1_macro": cv_f1,
               "test_accuracy": test["accuracy"], "test_f1_macro": test["f1_macro"],
               "test_correct": int(round(test["accuracy"] * len(yte))), "test_n": int(len(yte)),
               "fivefold_full_mean": r5_mean, "fivefold_full_std": r5_std,
               "tuning_seconds": time.perf_counter() - t0}
        out["results"].append(row)
        models[name] = {"model": model, "scalers": (mm, std)}
        print(json.dumps({k: row[k] for k in ["model_name", "cv_mean_accuracy", "test_accuracy", "test_correct"]}), flush=True)
        OUT_JSON.write_text(json.dumps(out, indent=2))

    with open(OUT_PKL, "wb") as fh:
        pickle.dump(models, fh)


if __name__ == "__main__":
    main()
