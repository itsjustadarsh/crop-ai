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
import torch
import torch.nn as nn
import torch.optim as optim
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

torch.set_num_threads(1)

OUTER_SEED = 42
INNER_CV_SEED = 42
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
DATASET_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets" / "atharvaingle" / "crop-recommendation-dataset" / "versions" / "1" / "Crop_recommendation.csv"
BASE = Path(__file__).resolve().parent.parent
ASSETS = BASE / "paper" / "assets"
MODEL_DIR = BASE / "model"
RESULTS_JSON = ASSETS / "final_study_results.json"
RESULTS_CSV = ASSETS / "final_study_results.csv"
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
    "olive": "#6b7a34",
    "tan": "#d6c29c",
}

SIMPLICITY_RANK = {
    "Extra Trees": 0,
    "Random Forest": 1,
    "SVM (RBF)": 2,
    "LightGBM": 3,
    "XGBoost": 4,
    "Weighted Voting": 5,
    "Stacking": 6,
    "TabNet": 7,
    "1D CNN": 8,
    "Hybrid DL+RF": 9,
}


def set_seed(seed: int = OUTER_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    family: str
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
    fold_accuracies: list[float]
    fold_f1_macro: list[float]

    def as_dict(self) -> dict:
        return {
            "model_name": self.name,
            "family": self.family,
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
            "fold_accuracies": json.dumps(self.fold_accuracies),
            "fold_f1_macro": json.dumps(self.fold_f1_macro),
        }


class CropCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.10),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class DenseExtractor(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head = nn.Linear(32, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


def inner_val_split(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=seed,
        stratify=y_train,
    )


def train_cnn(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, seed: int, n_classes: int) -> tuple[np.ndarray, CropCNN, float]:
    X_fit, X_val, y_fit, y_val = inner_val_split(X_train, y_train, seed)
    model = CropCNN(n_classes)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, dtype=torch.float32), torch.tensor(y_fit, dtype=torch.long)),
        batch_size=128,
        shuffle=True,
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32)
    best_acc = -1.0
    patience = 0
    best_state = None
    start = time.perf_counter()
    for _ in range(80):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_val_t).argmax(1).numpy()
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 12:
            break
    elapsed = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_eval_t).argmax(1).numpy()
    return preds, model, elapsed


def train_hybrid(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, seed: int, n_classes: int) -> tuple[np.ndarray, tuple[DenseExtractor, RandomForestClassifier], float]:
    X_fit, _, y_fit, _ = inner_val_split(X_train, y_train, seed)
    model = DenseExtractor(X_train.shape[1], n_classes)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_fit, dtype=torch.float32), torch.tensor(y_fit, dtype=torch.long)),
        batch_size=256,
        shuffle=True,
    )
    start = time.perf_counter()
    for _ in range(70):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        sched.step(epoch_loss)
    model.eval()
    with torch.no_grad():
        emb_train = model.embed(torch.tensor(X_train, dtype=torch.float32)).numpy()
        emb_eval = model.embed(torch.tensor(X_eval, dtype=torch.float32)).numpy()
    rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=1)
    rf.fit(emb_train, y_train)
    preds = rf.predict(emb_eval)
    elapsed = time.perf_counter() - start
    return preds, (model, rf), elapsed


def train_tabnet(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, seed: int) -> tuple[np.ndarray, TabNetClassifier, float]:
    X_fit, X_val, y_fit, y_val = inner_val_split(X_train, y_train, seed)
    clf = TabNetClassifier(
        n_d=24,
        n_a=24,
        n_steps=4,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        momentum=0.02,
        epsilon=1e-15,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-3),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=30, gamma=0.9),
        mask_type="sparsemax",
        verbose=0,
        seed=seed,
    )
    start = time.perf_counter()
    clf.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        max_epochs=100,
        patience=12,
        batch_size=256,
        virtual_batch_size=128,
    )
    elapsed = time.perf_counter() - start
    preds = clf.predict(X_eval)
    return preds, clf, elapsed


def build_candidates() -> list[tuple[str, str, object]]:
    return [
        ("Extra Trees", "classical_ml", lambda seed: ExtraTreesClassifier(n_estimators=1200, random_state=seed, n_jobs=1)),
        ("Random Forest", "classical_ml", lambda seed: RandomForestClassifier(n_estimators=700, random_state=seed, n_jobs=1)),
        ("SVM (RBF)", "classical_ml", lambda seed: SVC(kernel="rbf", C=20, probability=True, random_state=seed)),
        ("XGBoost", "classical_ml", lambda seed: XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.95, colsample_bytree=0.95, eval_metric="mlogloss", verbosity=0, random_state=seed, n_jobs=1)),
        ("LightGBM", "classical_ml", lambda seed: LGBMClassifier(objective="multiclass", num_class=22, n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.95, colsample_bytree=0.95, random_state=seed, verbosity=-1, n_jobs=1)),
        (
            "Stacking",
            "classical_ml",
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
        (
            "Weighted Voting",
            "classical_ml",
            lambda seed: VotingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=700, random_state=seed, n_jobs=1)),
                    ("et", ExtraTreesClassifier(n_estimators=1200, random_state=seed, n_jobs=1)),
                    ("xgb", XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.95, colsample_bytree=0.95, eval_metric="mlogloss", verbosity=0, random_state=seed, n_jobs=1)),
                    ("svc", SVC(kernel="rbf", C=20, probability=True, random_state=seed)),
                ],
                voting="soft",
                weights=[2, 3, 1, 1],
                n_jobs=1,
            ),
        ),
        ("1D CNN", "deep_learning", None),
        ("TabNet", "deep_learning", None),
        ("Hybrid DL+RF", "hybrid_ml_dl", None),
    ]


def fit_predict(name: str, family: str, factory, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, seed: int, labels_sorted: list[str]) -> tuple[np.ndarray, object, float]:
    if name == "1D CNN":
        return train_cnn(X_train, y_train, X_eval, seed, len(labels_sorted))
    if name == "TabNet":
        return train_tabnet(X_train, y_train, X_eval, seed)
    if name == "Hybrid DL+RF":
        return train_hybrid(X_train, y_train, X_eval, seed, len(labels_sorted))
    model = factory(seed)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    preds = model.predict(X_eval)
    return preds, model, elapsed


def prepare_variant(variant_name: str, X_train_raw: np.ndarray, y_train_raw: np.ndarray, X_test_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if variant_name == "raw":
        X_train_variant, y_train_variant = X_train_raw, y_train_raw
    else:
        X_train_variant, y_train_variant = augment_mvn(X_train_raw, y_train_raw, factor=3, seed=OUTER_SEED)
    mm, std = fit_scalers(X_train_variant)
    X_train_scaled, X_test_scaled = scale_pair(mm, std, X_train_variant, X_test_raw)
    return X_train_scaled, y_train_variant, X_test_scaled


def evaluate_candidate(name: str, family: str, factory, variant_name: str, X_train_raw: np.ndarray, y_train_raw: np.ndarray, X_test_raw: np.ndarray, y_test: np.ndarray, labels_sorted: list[str]) -> tuple[EvalResult, object]:
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
        preds, _, _ = fit_predict(name, family, factory, X_fold_train_s, y_fold_train, X_fold_val_s, OUTER_SEED + fold_idx, labels_sorted)
        metrics = metric_dict(y_fold_val, preds)
        fold_accs.append(metrics["accuracy"])
        fold_f1s.append(metrics["f1_macro"])

    X_train_scaled, y_train_variant, X_test_scaled = prepare_variant(variant_name, X_train_raw, y_train_raw, X_test_raw)
    test_preds, fitted_model, train_seconds = fit_predict(name, family, factory, X_train_scaled, y_train_variant, X_test_scaled, OUTER_SEED, labels_sorted)
    train_preds, _, _ = fit_predict(name, family, factory, X_train_scaled, y_train_variant, X_train_scaled, OUTER_SEED, labels_sorted)
    train_metrics = metric_dict(y_train_variant, train_preds)
    test_metrics = metric_dict(y_test, test_preds)

    result = EvalResult(
        name=name,
        family=family,
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
        fold_accuracies=[round(v, 6) for v in fold_accs],
        fold_f1_macro=[round(v, 6) for v in fold_f1s],
    )
    return result, fitted_model


def plot_feature_distribution(df: pd.DataFrame) -> None:
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


def plot_feature_heatmap(X_raw: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    corr = pd.DataFrame(X_raw, columns=FEATURES).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", square=True, ax=ax, cbar=True)
    ax.set_title("Feature Correlation Heatmap", color=P["text"], fontsize=14, fontweight="bold")
    ax.tick_params(colors=P["text"])
    plt.tight_layout()
    fig.savefig(ASSETS / "feature_heatmap.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)


def plot_accuracy_chart(results_df: pd.DataFrame, finalist_names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    plot_df = results_df[results_df["model_name"].isin(finalist_names)].copy()
    plot_df["model_name"] = pd.Categorical(plot_df["model_name"], categories=finalist_names, ordered=True)
    plot_df = plot_df.sort_values(["model_name", "dataset_variant"])
    sns.barplot(
        data=plot_df,
        x="model_name",
        y="test_accuracy",
        hue="dataset_variant",
        palette={"raw": P["blue"], "augmented": P["green"]},
        ax=ax,
    )
    best_model = plot_df[plot_df["dataset_variant"] == "raw"].sort_values(["cv_mean_accuracy", "test_accuracy"], ascending=False).iloc[0]["model_name"]
    for label in ax.get_xticklabels():
        if label.get_text() == best_model:
            label.set_color(P["amber"])
            label.set_fontweight("bold")
        else:
            label.set_color(P["text"])
    ax.set_xlabel("")
    ax.set_ylabel("Outer-Test Accuracy", color=P["text"])
    ax.set_ylim(0.95, 1.01)
    ax.tick_params(axis="y", colors=P["text"])
    ax.legend(title="", facecolor=P["panel"])
    ax.set_title("Final Model Set: Outer-Test Accuracy by Training Variant", color=P["text"], fontweight="bold")
    plt.tight_layout()
    fig.savefig(ASSETS / "final_model_accuracy.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)


def plot_confusion(cm: np.ndarray, labels: list[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    sns.heatmap(cm, cmap="YlGnBu", ax=ax, cbar=True)
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, color=P["text"])
    ax.set_yticklabels(labels, fontsize=7, color=P["text"])
    ax.set_title(title, color=P["text"], fontweight="bold")
    ax.set_xlabel("Predicted label", color=P["text"])
    ax.set_ylabel("True label", color=P["text"])
    plt.tight_layout()
    fig.savefig(ASSETS / "best_model_confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=P["bg"])
    ax.axis("off")
    blocks = [
        ("7 agronomic\nfeatures", 0.05, P["tan"]),
        ("Conv1d 1→64\nBN + ReLU", 0.24, P["blue"]),
        ("Conv1d 64→128\nBN + ReLU", 0.43, P["blue"]),
        ("Conv1d 128→64\nBN + ReLU", 0.62, P["olive"]),
        ("GAP + MLP\n64→128→22", 0.81, P["green"]),
    ]
    for label, x, color in blocks:
        rect = plt.Rectangle((x, 0.32), 0.14, 0.32, facecolor=color, edgecolor=P["border"], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.07, 0.48, label, ha="center", va="center", fontsize=10, color=P["text"], fontweight="bold")
    for idx in range(len(blocks) - 1):
        x0 = blocks[idx][1] + 0.14
        x1 = blocks[idx + 1][1]
        ax.annotate("", xy=(x1, 0.48), xytext=(x0, 0.48), arrowprops=dict(arrowstyle="->", color=P["sub"], lw=2))
    ax.text(0.69, 0.2, "GradCAM target layer", ha="center", fontsize=9, color=P["amber"], fontweight="bold")
    fig.savefig(ASSETS / "dl_architecture.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)


def repeated_cv_summary(name: str, family: str, factory, X_raw: np.ndarray, y_raw: np.ndarray, labels_sorted: list[str]) -> dict:
    if family != "classical_ml":
        return {}
    rkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    scores = []
    for split_idx, (tr_idx, te_idx) in enumerate(rkf.split(X_raw, y_raw), start=1):
        X_tr = X_raw[tr_idx]
        y_tr = y_raw[tr_idx]
        X_te = X_raw[te_idx]
        y_te = y_raw[te_idx]
        mm, std = fit_scalers(X_tr)
        X_tr_s, X_te_s = scale_pair(mm, std, X_tr, X_te)
        preds, _, _ = fit_predict(name, family, factory, X_tr_s, y_tr, X_te_s, OUTER_SEED + split_idx, labels_sorted)
        scores.append(float(accuracy_score(y_te, preds)))
    return {
        "robustness_cv_mean_accuracy": round(float(np.mean(scores)), 6),
        "robustness_cv_std_accuracy": round(float(np.std(scores)), 6),
        "robustness_cv_scores": [round(v, 6) for v in scores],
    }


def winner_sort_key(row: pd.Series) -> tuple:
    return (
        float(row["cv_mean_accuracy"]),
        float(row["cv_mean_f1_macro"]),
        -float(row["generalization_gap"]),
        -SIMPLICITY_RANK.get(row["model_name"], 999),
    )


def save_results(results_df: pd.DataFrame, payload: dict) -> None:
    results_df.to_csv(RESULTS_CSV, index=False)
    RESULTS_JSON.write_text(json.dumps(payload, indent=2))


def main() -> None:
    set_seed()
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
    plot_feature_heatmap(X_raw)
    plot_architecture()

    results = []
    trained_models = {}
    candidate_defs = build_candidates()

    for variant_name in ["raw", "augmented"]:
        for name, family, factory in candidate_defs:
            result, fitted_model = evaluate_candidate(name, family, factory, variant_name, X_train_raw, y_train_raw, X_test_raw, y_test, labels_sorted)
            results.append(result.as_dict())
            trained_models[(variant_name, name)] = fitted_model

    results_df = pd.DataFrame(results)
    raw_df = results_df[results_df["dataset_variant"] == "raw"].copy()
    raw_df = raw_df.sort_values(
        by=["cv_mean_accuracy", "cv_mean_f1_macro", "generalization_gap"],
        ascending=[False, False, True],
    )

    top_classical = raw_df[raw_df["family"] == "classical_ml"].head(3)
    top_deep = raw_df[raw_df["family"] == "deep_learning"].head(1)
    top_hybrid = raw_df[raw_df["family"] == "hybrid_ml_dl"].head(1)
    finalists = pd.concat([top_classical, top_deep, top_hybrid], ignore_index=True)
    finalists = finalists.sort_values(
        by=["cv_mean_accuracy", "cv_mean_f1_macro", "generalization_gap"],
        ascending=[False, False, True],
    )
    finalist_names = finalists["model_name"].tolist()

    best_row = sorted(raw_df.to_dict("records"), key=lambda row: (
        row["cv_mean_accuracy"],
        row["cv_mean_f1_macro"],
        -row["generalization_gap"],
        -SIMPLICITY_RANK.get(row["model_name"], 999),
    ), reverse=True)[0]
    best_name = best_row["model_name"]
    best_family = best_row["family"]

    plot_accuracy_chart(results_df, finalist_names)

    X_train_scaled, _, X_test_scaled = prepare_variant("raw", X_train_raw, y_train_raw, X_test_raw)
    best_model = trained_models[("raw", best_name)]
    if isinstance(best_model, tuple):
        extractor, rf = best_model
        extractor.eval()
        with torch.no_grad():
            emb = extractor.embed(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
        best_preds = rf.predict(emb)
    elif isinstance(best_model, nn.Module):
        best_model.eval()
        with torch.no_grad():
            best_preds = best_model(torch.tensor(X_test_scaled, dtype=torch.float32)).argmax(1).numpy()
    else:
        best_preds = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, best_preds)
    plot_confusion(cm, labels_sorted, f"Confusion Matrix: {best_name} on Outer Test Split")

    best_factory = next(factory for name, family, factory in candidate_defs if name == best_name)
    robustness = repeated_cv_summary(best_name, best_family, best_factory, X_raw, y_raw, labels_sorted)

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
        },
        "finalists": {
            "top_3_classical": top_classical["model_name"].tolist(),
            "best_deep_learning": top_deep["model_name"].tolist(),
            "best_hybrid": top_hybrid["model_name"].tolist(),
            "final_table_order": finalist_names,
        },
        "research_best_model": best_name,
        "results": results,
        "headline_model_robustness": robustness,
        "publication_note": "Final claims should use CV-guided winner selection plus outer-test evaluation, not split-picked exploratory maxima.",
    }
    save_results(results_df, payload)

    mm_raw, std_raw = fit_scalers(X_train_raw)
    with open(BEST_MODEL_PKL, "wb") as handle:
        pickle.dump(
            {
                "model_name": best_name,
                "family": best_family,
                "scalers": {"minmax": mm_raw, "standard": std_raw},
                "model": best_model,
                "labels": labels_sorted,
                "features": FEATURES,
            },
            handle,
        )

    best_row_df = raw_df[raw_df["model_name"] == best_name].iloc[0]
    meta = {
        "research_best_model": best_name,
        "family": best_family,
        "dataset_variant": "raw",
        "cv_mean_accuracy": round(float(best_row_df["cv_mean_accuracy"]), 6),
        "cv_std_accuracy": round(float(best_row_df["cv_std_accuracy"]), 6),
        "test_accuracy": round(float(best_row_df["test_accuracy"]), 6),
        "test_f1_macro": round(float(best_row_df["test_f1_macro"]), 6),
        "generalization_gap": round(float(best_row_df["generalization_gap"]), 6),
        "finalists": payload["finalists"],
        "dataset_path": str(DATASET_CACHE),
        "outer_split_seed": OUTER_SEED,
        "inner_cv_seed": INNER_CV_SEED,
        "selection_protocol": payload["selection_protocol"],
        "headline_model_robustness": robustness,
    }
    BEST_MODEL_META.write_text(json.dumps(meta, indent=2))

    print("Nested validation final study complete.")
    print(results_df.sort_values(["dataset_variant", "cv_mean_accuracy", "test_accuracy"], ascending=[True, False, False]).to_string(index=False))
    print(f"Research best model: {best_name}")
    print(f"CV mean accuracy: {best_row_df['cv_mean_accuracy']:.4f}")
    print(f"Outer-test accuracy: {best_row_df['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
