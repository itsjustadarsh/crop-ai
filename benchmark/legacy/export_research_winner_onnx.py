import json
import pickle
from pathlib import Path

import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "model"
BEST_MODEL_PKL = MODEL_DIR / "research_best_model.pkl"
BEST_MODEL_META = MODEL_DIR / "research_model_config.json"
MODEL_CONFIG = MODEL_DIR / "model_config.json"
ONNX_PATH = BASE / "crop_model.onnx"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
DATASET_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets" / "atharvaingle" / "crop-recommendation-dataset" / "versions" / "1" / "Crop_recommendation.csv"


def build_factory(name: str):
    factories = {
        "Random Forest": lambda: RandomForestClassifier(n_estimators=700, random_state=42, n_jobs=1),
        "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=1200, random_state=42, n_jobs=1),
        "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
        "SVM (RBF)": lambda: SVC(kernel="rbf", C=20, probability=True, random_state=42),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42),
        "Stacking": lambda: StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=1)),
                ("et", ExtraTreesClassifier(n_estimators=700, random_state=42, n_jobs=1)),
                ("svc", SVC(kernel="rbf", C=20, probability=True, random_state=42)),
                ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42)),
            ],
            final_estimator=LogisticRegression(max_iter=2000, random_state=42),
            cv=3,
            n_jobs=1,
            stack_method="predict_proba",
        ),
    }
    return factories[name]


def main():
    meta = json.loads(BEST_MODEL_META.read_text())
    winner_name = meta["research_best_model"]
    factory = build_factory(winner_name)
    df = pd.read_csv(DATASET_CACHE)
    labels_sorted = sorted(df["label"].unique())
    X = df[FEATURES].to_numpy(dtype="float32")
    y = df["label"].to_numpy()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mm = MinMaxScaler()
    std = StandardScaler()
    X_train_mm = mm.fit_transform(X_train)
    X_train_scaled = std.fit_transform(X_train_mm)
    model = factory()
    model.fit(X_train_scaled, y_train)
    payload = {
        "model_name": winner_name,
        "family": "classical_ml",
        "scalers": {"minmax": mm, "standard": std},
        "model": model,
        "labels": labels_sorted,
        "features": FEATURES,
    }
    with open(BEST_MODEL_PKL, "wb") as handle:
        pickle.dump(payload, handle)
    pipeline = Pipeline(
        [
            ("minmax", payload["scalers"]["minmax"]),
            ("standard", payload["scalers"]["standard"]),
            ("clf", payload["model"]),
        ]
    )
    onx = convert_sklearn(
        pipeline,
        initial_types=[("float_input", FloatTensorType([None, len(payload["features"])]))],
        options={id(pipeline): {"zipmap": False}},
    )
    ONNX_PATH.write_bytes(onx.SerializeToString())

    model_cfg = {
        "model_type": f'{payload["model_name"]} with embedded MinMax+Standard scaling — ONNX',
        "feature_names": payload["features"],
        "num_features": len(payload["features"]),
        "num_classes": len(payload["labels"]),
        "crop_mapping": {label: label for label in payload["labels"]},
        "model_classes_type": "string",
        "research_best_model": payload["model_name"],
        "notes": "This ONNX artifact is the paper winner model and the deployed model.",
    }
    MODEL_CONFIG.write_text(json.dumps(model_cfg, indent=2))
    print(json.dumps({"onnx_path": str(ONNX_PATH), "winner": payload["model_name"]}, indent=2))


if __name__ == "__main__":
    main()
