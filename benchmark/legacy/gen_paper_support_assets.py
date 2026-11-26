import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-crop-ai")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

BASE = Path(__file__).resolve().parent.parent
ASSETS = BASE / "paper" / "assets"
MODEL_DIR = BASE / "model"
RESULTS_JSON = ASSETS / "onnx_winner_results.json"
BEST_MODEL_PKL = MODEL_DIR / "research_best_model.pkl"

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


def plot_architecture():
    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor=P["bg"])
    ax.set_facecolor(P["bg"])
    ax.axis("off")
    blocks = [
        (0.06, 0.32, 0.18, 0.36, P["amber"], "Agent 1\nSoil Analyzer\nSoilGrids 2.0"),
        (0.29, 0.32, 0.18, 0.36, P["green"], "Agent 2\nWeather Analyst\nOpen-Meteo"),
        (0.52, 0.32, 0.18, 0.36, P["blue"], "Agent 3\nCrop Predictor\nWinner ONNX"),
        (0.75, 0.32, 0.18, 0.36, P["red"], "Agent 4\nInsight Engine\nLLM Report"),
    ]
    for x, y, w, h, color, text in blocks:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=P["border"], linewidth=1.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=12, fontweight="bold", transform=ax.transAxes)
    for start, end in [(0.24, 0.29), (0.47, 0.52), (0.70, 0.75)]:
        ax.annotate("", xy=(end, 0.5), xytext=(start, 0.5), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color=P["sub"], linewidth=2))
    ax.text(0.15, 0.18, "GPS + location", ha="center", color=P["sub"], fontsize=10, transform=ax.transAxes)
    ax.text(0.38, 0.18, "temperature + humidity + rainfall", ha="center", color=P["sub"], fontsize=10, transform=ax.transAxes)
    ax.text(0.61, 0.18, "recommended crop", ha="center", color=P["sub"], fontsize=10, transform=ax.transAxes)
    ax.text(0.84, 0.18, "agronomic explanation", ha="center", color=P["sub"], fontsize=10, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(ASSETS / "llm_architecture.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)


def plot_literature_comparison():
    payload = json.loads(RESULTS_JSON.read_text())
    ours = next(row for row in payload["results"] if row["model_name"] == payload["research_best_model"] and row["dataset_variant"] == "raw")["test_accuracy"] * 100.0
    labels = ["Crop AI\n(ours)", "Alam et al.\n2025", "Prity et al.\n2024", "Shastri et al.\n2025"]
    values = [ours, 99.54, 99.31, 99.27]
    colors = [P["blue"], P["green"], P["green"], P["green"]]
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    bars = ax.bar(labels, values, color=colors, edgecolor=P["border"], linewidth=1.0)
    ax.set_ylim(99.15, 99.7)
    ax.set_ylabel("Reported Accuracy (%)", color=P["text"])
    ax.set_title("Crop AI vs Lower-Accuracy Same-Dataset Studies", color=P["text"], fontweight="bold")
    ax.tick_params(axis="x", labelcolor=P["text"])
    ax.tick_params(axis="y", colors=P["text"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", va="bottom", color=P["text"], fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "literature_comparison.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)


def plot_lime_and_shap():
    payload = pickle.load(open(BEST_MODEL_PKL, "rb"))
    features = payload["features"]
    labels = payload["labels"]
    model = payload["model"]
    mm = payload["scalers"]["minmax"]
    std = payload["scalers"]["standard"]

    import pandas as pd
    from sklearn.model_selection import train_test_split
    data = pd.read_csv(Path.home() / ".cache" / "kagglehub" / "datasets" / "atharvaingle" / "crop-recommendation-dataset" / "versions" / "1" / "Crop_recommendation.csv")
    X = data[features].to_numpy(dtype=np.float32)
    y = data["label"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_s = std.transform(mm.transform(X_train))
    X_test_s = std.transform(mm.transform(X_test))

    # LIME on the first correctly predicted rice example, or fallback to first correct sample.
    preds = model.predict(X_test_s)
    idx = None
    for i, (yt, yp) in enumerate(zip(y_test, preds)):
        if yt == yp and yt == "rice":
            idx = i
            break
    if idx is None:
        idx = next(i for i, (yt, yp) in enumerate(zip(y_test, preds)) if yt == yp)
    explainer = LimeTabularExplainer(X_train_s, feature_names=features, class_names=labels, discretize_continuous=True)
    exp = explainer.explain_instance(X_test_s[idx], model.predict_proba, num_features=len(features), top_labels=1)
    label_idx = list(model.classes_).index(preds[idx])
    pred_label = preds[idx]
    lime_pairs = exp.as_list(label=label_idx)
    names = [name for name, _ in lime_pairs][::-1]
    vals = [val for _, val in lime_pairs][::-1]
    colors = [P["green"] if v > 0 else P["red"] for v in vals]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=P["bg"])
    ax.set_facecolor(P["panel"])
    ax.barh(names, vals, color=colors, edgecolor=P["border"])
    ax.set_title(f"LIME Explanation for {pred_label.title()} Prediction", color=P["text"], fontweight="bold")
    ax.tick_params(colors=P["text"])
    ax.axvline(0, color=P["sub"], linewidth=1)
    fig.tight_layout()
    fig.savefig(ASSETS / "lime_explanation.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)

    # SHAP for tree-based winner.
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(X_test_s[:120])
    class_idx = list(model.classes_).index(preds[idx])
    class_shap = shap_values[class_idx] if isinstance(shap_values, list) else shap_values[:, :, class_idx]
    plt.figure(figsize=(9, 5), facecolor=P["bg"])
    shap.summary_plot(class_shap, X_test_s[:120], feature_names=features, show=False)
    plt.gcf().savefig(ASSETS / "shap_summary.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close()

    sample_shap = class_shap[idx]
    shap.plots._waterfall.waterfall_legacy(explainer_shap.expected_value[class_idx] if isinstance(explainer_shap.expected_value, (list, np.ndarray)) else explainer_shap.expected_value, sample_shap, feature_names=features, show=False)
    plt.gcf().savefig(ASSETS / "shap_waterfall_rice.png", dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close()


def main():
    plot_architecture()
    plot_literature_comparison()
    plot_lime_and_shap()


if __name__ == "__main__":
    main()
