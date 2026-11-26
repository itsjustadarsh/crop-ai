import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-crop-ai")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

OUTER_SEED = 42
INNER_CV_SEED = 42
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
DATASET_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets" / "atharvaingle" / "crop-recommendation-dataset" / "versions" / "1" / "Crop_recommendation.csv"
BASE = Path(__file__).resolve().parent.parent
ASSETS = BASE / "paper" / "assets"
MODEL_DIR = BASE / "model"
RESULTS_JSON = ASSETS / "onnx_winner_results.json"
RESULTS_CSV = ASSETS / "onnx_winner_results.csv"
BEST_MODEL_PKL = MODEL_DIR / "research_best_model.pkl"
BEST_MODEL_META = MODEL_DIR / "research_model_config.json"

ASSETS.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

P = {
    "bg": "#ffffff",
    "panel": "#f7f4ea",
    "border": "#d2c6a8",
    "text": "#1d1a16",
    "sub": "#5d5447",
    "green": "#3f7d20",
    "blue": "#2f6e84",
    "amber": "#bf7a1f",
    "red": "#b64926",
}

SIMPLICITY_RANK = {
    "Random Forest": 0,
    "Extra Trees": 1,
    "Gradient Boosting": 2,
    "SVM (RBF)": 3,
    "MLP": 4,
    "Stacking": 5,
}


def load_dataset() -> pd.DataFrame:
    if not DATASET_CACHE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_CACHE}")
    df = pd.read_csv(DATASET_CACHE)
    labels_sorted = sorted(df["label"].unique())
    df["label_id"] = df["label"].map({name: idx for idx, name in enumerate(labels_sorted)})
    return df


def augment_mvn(X: np.ndarray, y: np.ndarray, factor: int = 3, seed: int = OUTER_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_parts = [X]
    y_parts = [y]
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    for cls in np.unique(y):
        Xc = X[y == cls]
        mu = Xc.mean(axis=0)
        cov = np.cov(Xc.T) + np.eye(X.shape[1]) * 1e-6
        n_new = len(Xc) * (factor - 1)
        Xn = rng.multivariate_normal(mu, cov, size=n_new).astype(np.float32)
        Xn = np.clip(Xn, mins, maxs)
        X_parts.append(Xn)
        y_parts.append(np.full(n_new, cls, dtype=np.int64))
    return np.vstack(X_parts), np.concatenate(y_parts)


def fit_scalers(X_train: np.ndarray) -> tuple[MinMaxScaler, StandardScaler]:
    mm = MinMaxScaler()
    std = StandardScaler()
    X_train_mm = mm.fit_transform(X_train)
    std.fit(X_train_mm)
    return mm, std


def scale_pair(mm: MinMaxScaler, std: StandardScaler, X_train: np.ndarray, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return std.transform(mm.transform(X_train)), std.transform(mm.transform(X_eval))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@dataclass
class EvalResult:
    name: str
    variant: str
    train_accuracy: float
    train_f1_macro: float
    cv_mean_accuracy: float
    cv_std_accuracy: float
    cv_mean_f1_macro: float
    cv_std_f1_macro: float
    test_accuracy: float
    test_precision_macro: float
    test_recall_macro: float
    test_f1_macro: float
    generalization_gap: float
    train_seconds: float

    def as_dict(self) -> dict:
        return {
            "model_name": self.name,
            "dataset_variant": self.variant,
            "train_accuracy": self.train_accuracy,
            "train_f1_macro": self.train_f1_macro,
            "cv_mean_accuracy": self.cv_mean_accuracy,
            "cv_std_accuracy": self.cv_std_accuracy,
            "cv_mean_f1_macro": self.cv_mean_f1_macro,
            "cv_std_f1_macro": self.cv_std_f1_macro,
            "test_accuracy": self.test_accuracy,
            "test_precision_macro": self.test_precision_macro,
            "test_recall_macro": self.test_recall_macro,
            "test_f1_macro": self.test_f1_macro,
            "generalization_gap": self.generalization_gap,
            "train_seconds": self.train_seconds,
        }


def build_candidates():
    return [
        ("Random Forest", lambda seed: RandomForestClassifier(n_estimators=700, random_state=seed, n_jobs=1)),
        ("Extra Trees", lambda seed: ExtraTreesClassifier(n_estimators=1200, random_state=seed, n_jobs=1)),
        ("Gradient Boosting", lambda seed: GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=seed)),
        ("SVM (RBF)", lambda seed: SVC(kernel="rbf", C=20, probability=True, random_state=seed)),
        ("MLP", lambda seed: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=seed)),
        (
            "Stacking",
            lambda seed: StackingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=1)),
                    ("et", ExtraTreesClassifier(n_estimators=700, random_state=seed, n_jobs=1)),
                    ("svc", SVC(kernel="rbf", C=20, probability=True, random_state=seed)),
                    ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=seed)),
                ],
                final_estimator=LogisticRegression(max_iter=2000, random_state=seed),
                cv=3,
                n_jobs=1,
                stack_method="predict_proba",
            ),
        ),
    ]


def prepare_variant(variant_name: str, X_train_raw: np.ndarray, y_train_raw: np.ndarray, X_test_raw: np.ndarray):
    if variant_name == "raw":
        X_train_variant, y_train_variant = X_train_raw, y_train_raw
    else:
        X_train_variant, y_train_variant = augment_mvn(X_train_raw, y_train_raw, factor=3, seed=OUTER_SEED)
    mm, std = fit_scalers(X_train_variant)
    X_train_scaled, X_test_scaled = scale_pair(mm, std, X_train_variant, X_test_raw)
    return X_train_variant, y_train_variant, mm, std, X_train_scaled, X_test_scaled


def fit_model(factory, X_train: np.ndarray, y_train: np.ndarray, seed: int):
    model = factory(seed)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    return model, elapsed


def evaluate_candidate(name, factory, variant_name, X_train_raw, y_train_raw, X_test_raw, y_test):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=INNER_CV_SEED)
    fold_accs = []
    fold_f1s = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train_raw, y_train_raw), start=1):
        X_fold_train = X_train_raw[tr_idx]
        y_fold_train = y_train_raw[tr_idx]
        X_fold_val = X_train_raw[va_idx]
        y_fold_val = y_train_raw[va_idx]
        if variant_name == "augmented":
            X_fold_train, y_fold_train = augment_mvn(X_fold_train, y_fold_train, factor=3, seed=OUTER_SEED + fold_idx)
        mm, std = fit_scalers(X_fold_train)
        X_fold_train_s, X_fold_val_s = scale_pair(mm, std, X_fold_train, X_fold_val)
        model, _ = fit_model(factory, X_fold_train_s, y_fold_train, OUTER_SEED + fold_idx)
        preds = model.predict(X_fold_val_s)
        metrics = metric_dict(y_fold_val, preds)
        fold_accs.append(metrics["accuracy"])
        fold_f1s.append(metrics["f1_macro"])

    X_train_variant, y_train_variant, mm, std, X_train_scaled, X_test_scaled = prepare_variant(variant_name, X_train_raw, y_train_raw, X_test_raw)
    model, train_seconds = fit_model(factory, X_train_scaled, y_train_variant, OUTER_SEED)
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    train_metrics = metric_dict(y_train_variant, train_preds)
    test_metrics = metric_dict(y_test, test_preds)

    result = EvalResult(
        name=name,
        variant=variant_name,
        train_accuracy=train_metrics["accuracy"],
        train_f1_macro=train_metrics["f1_macro"],
        cv_mean_accuracy=float(np.mean(fold_accs)),
        cv_std_accuracy=float(np.std(fold_accs)),
        cv_mean_f1_macro=float(np.mean(fold_f1s)),
        cv_std_f1_macro=float(np.std(fold_f1s)),
        test_accuracy=test_metrics["accuracy"],
        test_precision_macro=test_metrics["precision_macro"],
        test_recall_macro=test_metrics["recall_macro"],
        test_f1_macro=test_metrics["f1_macro"],
        generalization_gap=float(train_metrics["accuracy"] - test_metrics["accuracy"]),
        train_seconds=train_seconds,
    )
    fitted = {
        "model": model,
        "scalers": {"minmax": mm, "standard": std},
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_variant": y_train_variant,
        "test_preds": test_preds,
    }
    return result, fitted


def plot_feature_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), facecolor=P["bg"])
    fig.suptitle("Feature Distribution by Crop", fontsize=16, color=P["text"], fontweight="bold")
    crop_names = sorted(df["label"].unique())
    colors = plt.cm.terrain(np.linspace(0.1, 0.9, len(crop_names)))
    for idx, feature in enumerate(FEATURES):
        ax = axes[idx // 4][idx % 4]
        ax.set_facecolor(P["panel"])
        values = [df[df["label"] == crop][feature].values for crop in crop_names]
        bp = ax.boxplot(values, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set(facecolor=color, alpha=0.75, edgecolor=P["border"])
        for key in ["whiskers", "caps", "medians"]:
            for artist in bp[key]:
                artist.set(color=P["sub"])
        ax.set_xticks(range(1, len(crop_names) + 1))
        ax.set_xticklabels([name[:4] for name in crop_names], rotation=90, fontsize=6, color=P["text"])
        ax.tick_params(colors=P["text"])
        ax.set_title(feature, color=P["text"], fontsize=10, fontweight="bold")
        ax.grid(axis="y", color=P["border"], linewidth=0.4)
    axes[1][3].set_visible(False)
    plt.tight_layout()
    fig.savefig(ASSETS / "feature_distribution.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)


def plot_feature_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    corr = df[FEATURES].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", square=True, ax=ax, cbar=True)
    ax.set_title("Feature Correlation Heatmap", color=P["text"], fontsize=14, fontweight="bold")
    ax.tick_params(colors=P["text"])
    plt.tight_layout()
    fig.savefig(ASSETS / "feature_heatmap.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)


def plot_accuracy_chart(results_df: pd.DataFrame, order: list[str]):
    plot_df = results_df[results_df["dataset_variant"] == "raw"].copy()
    plot_df["model_name"] = pd.Categorical(plot_df["model_name"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("model_name")
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    bars = ax.bar(plot_df["model_name"], plot_df["test_accuracy"] * 100.0, color=[P["green"] if name == order[0] else P["blue"] for name in plot_df["model_name"]], edgecolor=P["border"])
    ax.set_ylim(97.5, 100.0)
    ax.set_ylabel("Outer-Test Accuracy (%)", color=P["text"])
    ax.set_title("Validated Accuracy of ONNX-Friendly Candidate Models", color=P["text"], fontweight="bold")
    ax.tick_params(axis="x", colors=P["text"], rotation=20)
    ax.tick_params(axis="y", colors=P["text"])
    for bar, value in zip(bars, plot_df["test_accuracy"] * 100.0):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", color=P["text"], fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(ASSETS / "final_model_accuracy.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)


def plot_synthesis_accuracy(results_df: pd.DataFrame, order: list[str]):
    raw_df = results_df[results_df["dataset_variant"] == "raw"].set_index("model_name")
    aug_df = results_df[results_df["dataset_variant"] == "augmented"].set_index("model_name")
    before = [raw_df.loc[name, "test_accuracy"] * 100.0 for name in order]
    after = [aug_df.loc[name, "test_accuracy"] * 100.0 for name in order]
    x = np.arange(len(order))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    ax.bar(x - w / 2, before, width=w, color=P["blue"], label="Raw")
    ax.bar(x + w / 2, after, width=w, color=P["green"], label="Augmented")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right", color=P["text"])
    ax.tick_params(axis="y", colors=P["text"])
    ax.set_ylabel("Outer-Test Accuracy (%)", color=P["text"])
    ax.set_ylim(90.0, 100.0)
    ax.set_title("Accuracy Before and After Training-Only Augmentation", color=P["text"], fontweight="bold")
    ax.legend(facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"])
    plt.tight_layout()
    fig.savefig(ASSETS / "synthesis_accuracy_chart.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)


def plot_confusion(cm: np.ndarray, labels_sorted: list[str], winner_name: str):
    fig, ax = plt.subplots(figsize=(13, 11), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    sns.heatmap(cm, annot=True, fmt="d", linewidths=0.3, cmap="YlGnBu", ax=ax, xticklabels=labels_sorted, yticklabels=labels_sorted, cbar=True)
    ax.set_title(f"Confusion Matrix: {winner_name}", color=P["text"], fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", color=P["text"])
    ax.set_ylabel("True", color=P["text"])
    ax.tick_params(axis="x", colors=P["text"], rotation=45, labelsize=8)
    ax.tick_params(axis="y", colors=P["text"], rotation=0, labelsize=8)
    plt.tight_layout()
    fig.savefig(ASSETS / "best_model_confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)


def repeated_cv_summary(factory, X_raw, y_raw):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    scores = []
    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X_raw, y_raw), start=1):
        X_tr, y_tr = X_raw[tr_idx], y_raw[tr_idx]
        X_te, y_te = X_raw[te_idx], y_raw[te_idx]
        mm, std = fit_scalers(X_tr)
        X_tr_s, X_te_s = scale_pair(mm, std, X_tr, X_te)
        model, _ = fit_model(factory, X_tr_s, y_tr, OUTER_SEED + fold_idx)
        scores.append(float(accuracy_score(y_te, model.predict(X_te_s))))
    return {
        "robustness_cv_mean_accuracy": round(float(np.mean(scores)), 6),
        "robustness_cv_std_accuracy": round(float(np.std(scores)), 6),
        "robustness_cv_scores": [round(score, 6) for score in scores],
    }


def main():
    df = load_dataset()
    labels_sorted = sorted(df["label"].unique())
    X_raw = df[FEATURES].to_numpy(dtype=np.float32)
    y_raw = df["label_id"].to_numpy(dtype=np.int64)
    X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=0.2,
        random_state=OUTER_SEED,
        stratify=y_raw,
    )

    plot_feature_distribution(df)
    plot_feature_heatmap(df)

    results = []
    trained_models = {}
    candidate_defs = build_candidates()
    for variant_name in ["raw", "augmented"]:
        for name, factory in candidate_defs:
            result, fitted = evaluate_candidate(name, factory, variant_name, X_train_raw, y_train_raw, X_test_raw, y_test)
            results.append(result.as_dict())
            trained_models[(variant_name, name)] = fitted

    results_df = pd.DataFrame(results)
    raw_df = results_df[results_df["dataset_variant"] == "raw"].copy()
    raw_df = raw_df.sort_values(by=["cv_mean_accuracy", "cv_mean_f1_macro", "generalization_gap"], ascending=[False, False, True])
    order = raw_df["model_name"].tolist()
    best_row = raw_df.iloc[0].to_dict()
    best_name = best_row["model_name"]
    best_factory = next(factory for name, factory in candidate_defs if name == best_name)

    plot_accuracy_chart(results_df, order)
    plot_synthesis_accuracy(results_df, order)
    best_fitted = trained_models[("raw", best_name)]
    cm = confusion_matrix(y_test, best_fitted["test_preds"])
    plot_confusion(cm, labels_sorted, best_name)

    with open(BEST_MODEL_PKL, "wb") as handle:
        pickle.dump(
            {
                "model_name": best_name,
                "family": "classical_ml",
                "scalers": best_fitted["scalers"],
                "model": best_fitted["model"],
                "labels": labels_sorted,
                "features": FEATURES,
            },
            handle,
        )

    robustness = repeated_cv_summary(best_factory, X_raw, y_raw)
    payload = {
        "dataset": {
            "path": str(DATASET_CACHE),
            "original_samples": int(len(df)),
            "num_classes": int(len(labels_sorted)),
            "features": FEATURES,
            "outer_split_seed": OUTER_SEED,
            "inner_cv_seed": INNER_CV_SEED,
        },
        "selection_protocol": {
            "outer_split": "stratified holdout, test_size=0.2, random_state=42",
            "inner_selection": "3-fold stratified cross-validation on outer-train only",
            "augmentation_policy": "augmentation applied to training folds only; test split always raw and untouched",
            "ranking_metric": "cv_mean_accuracy",
            "tiebreakers": ["cv_mean_f1_macro", "smaller_generalization_gap", "simpler_model"],
            "candidate_pool": [name for name, _ in candidate_defs],
        },
        "research_best_model": best_name,
        "results": results,
        "headline_model_robustness": robustness,
        "publication_note": "Final paper uses the best ONNX-exportable model under the corrected validation protocol.",
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    results_df.to_csv(RESULTS_CSV, index=False)
    BEST_MODEL_META.write_text(
        json.dumps(
            {
                "research_best_model": best_name,
                "family": "classical_ml",
                "dataset_variant": "raw",
                "cv_mean_accuracy": round(float(best_row["cv_mean_accuracy"]), 6),
                "cv_std_accuracy": round(float(best_row["cv_std_accuracy"]), 6),
                "test_accuracy": round(float(best_row["test_accuracy"]), 6),
                "test_f1_macro": round(float(best_row["test_f1_macro"]), 6),
                "generalization_gap": round(float(best_row["generalization_gap"]), 6),
                "dataset_path": str(DATASET_CACHE),
                "selection_protocol": payload["selection_protocol"],
                "headline_model_robustness": robustness,
            },
            indent=2,
        )
    )
    print(json.dumps({"winner": best_name, "cv_mean_accuracy": best_row["cv_mean_accuracy"], "test_accuracy": best_row["test_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
