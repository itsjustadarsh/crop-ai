"""
Model performance benchmark — accuracy, per-class metrics, and visualisation.

Generates:
  • confusion_matrix.png      — 22×22 heatmap (PKL vs ONNX side-by-side)
  • per_class_metrics.png     — precision / recall / F1 grouped bar chart
  • feature_importance.png    — RandomForest feature importances
  • metrics_summary.png       — headline numbers card

Run from the benchmark/ directory:
    python model_benchmark.py
"""

import json
import pickle
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Paths (relative to this file) ────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MODEL_DIR  = ROOT / "model"
ONNX_PATH  = ROOT / "crop_model.onnx"

PKL_MODEL  = MODEL_DIR / "model.pkl"
PKL_MINMAX = MODEL_DIR / "minmaxscaler.pkl"
PKL_STD    = MODEL_DIR / "standscaler.pkl"
CFG_PATH   = MODEL_DIR / "model_config.json"
OUT_DIR    = Path(__file__).parent   # images saved alongside this script

# ── Palette ───────────────────────────────────────────────────────────────────
P = {
    "bg":      "#0a0a0a",
    "panel":   "#111111",
    "border":  "#1e1e1e",
    "text":    "#e0e0e0",
    "sub":     "#888888",
    "green":   "#22c55e",
    "blue":    "#3b82f6",
    "amber":   "#f59e0b",
    "red":     "#ef4444",
    "purple":  "#a855f7",
}

# ── 1. Config ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("CROP AI — MODEL PERFORMANCE BENCHMARK")
print("=" * 60)

with open(CFG_PATH) as f:
    cfg = json.load(f)

crop_map     = {int(k): v for k, v in cfg["crop_mapping"].items()}
ids_sorted   = sorted(crop_map)
names_sorted = [crop_map[i] for i in ids_sorted]
name_to_id   = {v: k for k, v in crop_map.items()}
FEATURES     = cfg["feature_names"]

# ── 2. Load models ────────────────────────────────────────────────────────────
print("\n[1/4] Loading PKL pipeline…")
model    = pickle.load(open(PKL_MODEL,  "rb"))
minmax   = pickle.load(open(PKL_MINMAX, "rb"))
std_sc   = pickle.load(open(PKL_STD,    "rb"))
pipeline = Pipeline([("minmax", minmax), ("standard", std_sc), ("clf", model)])
print("      OK")

print("[2/4] Loading ONNX session…")
import onnxruntime as ort
sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
print("      OK")

# ── 3. Evaluation data ────────────────────────────────────────────────────────
print("[3/4] Loading evaluation dataset…")

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    import pandas as pd

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "atharvaingle/crop-recommendation-dataset",
        "Crop_recommendation.csv",
    )
    X_raw  = df[FEATURES].values.astype(np.float32)
    y_true = np.array([name_to_id[lbl] for lbl in df["label"]])
    print(f"      Kaggle dataset loaded ({len(df)} rows)")

except Exception as e:
    print(f"      Kaggle unavailable: {e}")
    print("      Falling back to synthetic set from config ranges…")
    sp  = cfg["scaler_params"]
    mn  = np.array(sp["minmax_data_min"])
    mx  = np.array(sp["minmax_data_max"])
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for cid in ids_sorted:
        samples = rng.uniform(mn, mx, size=(100, len(FEATURES))).astype(np.float32)
        rows.append(samples)
        labels.extend([cid] * 100)
    X_raw  = np.vstack(rows)
    y_true = np.array(labels)
    print(f"      Synthetic fallback: {len(X_raw)} samples (100 per crop)")

# ── 4. Predict ────────────────────────────────────────────────────────────────
print("[4/4] Running predictions…")

# PKL — returns string crop names
y_pkl_raw = pipeline.predict(X_raw)
y_pkl = np.array(
    [name_to_id[n] if isinstance(n, str) else int(n) for n in y_pkl_raw]
)

# ONNX
y_onnx_raw = sess.run(None, {"float_input": X_raw})[0]
y_onnx = np.array(
    [name_to_id[str(n)] if isinstance(n, (str, np.str_)) else int(n)
     for n in y_onnx_raw]
)

# ── 5. Metrics ────────────────────────────────────────────────────────────────
acc_pkl  = accuracy_score(y_true, y_pkl)
acc_onnx = accuracy_score(y_true, y_onnx)

rep_pkl  = classification_report(
    y_true, y_pkl, labels=ids_sorted, target_names=names_sorted,
    output_dict=True, zero_division=0,
)
rep_onnx = classification_report(
    y_true, y_onnx, labels=ids_sorted, target_names=names_sorted,
    output_dict=True, zero_division=0,
)
cm_pkl   = confusion_matrix(y_true, y_pkl,  labels=ids_sorted)
cm_onnx  = confusion_matrix(y_true, y_onnx, labels=ids_sorted)

# ── Console table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'Metric':<35} {'PKL':>12} {'ONNX':>12}")
print("=" * 70)
for label, key in [
    ("Overall accuracy",    None),
    ("Macro avg precision", "precision"),
    ("Macro avg recall",    "recall"),
    ("Macro avg F1",        "f1-score"),
]:
    if key is None:
        vp, vo = acc_pkl, acc_onnx
    else:
        vp = rep_pkl["macro avg"][key]
        vo = rep_onnx["macro avg"][key]
    print(f"{label:<35} {vp:>12.4f} {vo:>12.4f}")
print("=" * 70)
print(f"\n{'Crop':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
print("-" * 60)
for n in names_sorted:
    r = rep_pkl.get(n, {})
    print(f"{n:<20} {r.get('precision',0):>10.3f} {r.get('recall',0):>8.3f}"
          f" {r.get('f1-score',0):>8.3f} {r.get('support',0):>9.0f}")
print("=" * 70)

# ── Helper ────────────────────────────────────────────────────────────────────
def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(P["panel"])
    ax.tick_params(colors=P["text"], labelsize=9)
    ax.xaxis.label.set_color(P["text"])
    ax.yaxis.label.set_color(P["text"])
    ax.title.set_color(P["text"])
    for sp in ax.spines.values():
        sp.set_edgecolor(P["border"])
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)

# ── Image 1: Confusion Matrix ─────────────────────────────────────────────────
print("\nGenerating confusion_matrix.png…")

fig, axes = plt.subplots(1, 2, figsize=(24, 11), facecolor=P["bg"])
fig.suptitle("Confusion Matrix — PKL vs ONNX", color=P["text"],
             fontsize=14, fontweight="bold", y=1.01)

for ax, cm, title in zip(axes, [cm_pkl, cm_onnx], ["PKL Pipeline", "ONNX Runtime"]):
    ax.set_facecolor(P["panel"])
    sns.heatmap(
        cm, annot=True, fmt="d", linewidths=0.4, cmap="Blues", ax=ax,
        xticklabels=names_sorted, yticklabels=names_sorted,
        cbar_kws={"shrink": 0.8}, annot_kws={"size": 7},
    )
    ax.set_title(title, color=P["text"], fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Predicted", color=P["text"], fontsize=9)
    ax.set_ylabel("True",      color=P["text"], fontsize=9)
    ax.tick_params(colors=P["text"], labelsize=8)
    ax.xaxis.set_tick_params(rotation=45)
    ax.yaxis.set_tick_params(rotation=0)
    ax.collections[0].colorbar.ax.tick_params(colors=P["text"])

plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight",
            facecolor=P["bg"])
plt.close()
print("      Saved.")

# ── Image 2: Per-class Precision / Recall / F1 ────────────────────────────────
print("Generating per_class_metrics.png…")

prec = [rep_pkl[n]["precision"]  for n in names_sorted]
rec  = [rep_pkl[n]["recall"]     for n in names_sorted]
f1   = [rep_pkl[n]["f1-score"]   for n in names_sorted]

x     = np.arange(len(names_sorted))
w     = 0.26

fig, ax = plt.subplots(figsize=(20, 7), facecolor=P["bg"])
style_ax(ax, title="Per-Class Metrics (PKL Pipeline)",
         xlabel="Crop", ylabel="Score")

ax.bar(x - w, prec, w, color=P["blue"],   alpha=0.9, label="Precision")
ax.bar(x,     rec,  w, color=P["green"],  alpha=0.9, label="Recall")
ax.bar(x + w, f1,   w, color=P["purple"], alpha=0.9, label="F1-Score")

ax.set_xticks(x)
ax.set_xticklabels(names_sorted, rotation=45, ha="right",
                   fontsize=8, color=P["text"])
ax.set_ylim(0, 1.08)
ax.axhline(acc_pkl, color=P["amber"], linewidth=1.2, linestyle="--",
           label=f"Overall accuracy {acc_pkl:.4f}")
ax.legend(facecolor=P["panel"], edgecolor=P["border"],
          labelcolor=P["text"], fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

plt.tight_layout()
plt.savefig(OUT_DIR / "per_class_metrics.png", dpi=150, bbox_inches="tight",
            facecolor=P["bg"])
plt.close()
print("      Saved.")

# ── Image 3: Feature Importance ───────────────────────────────────────────────
print("Generating feature_importance.png…")

importances = model.feature_importances_
order       = np.argsort(importances)[::-1]
feat_names  = [FEATURES[i] for i in order]
feat_vals   = importances[order]
bar_colors  = [P["green"], P["blue"], P["purple"], P["amber"],
               P["red"],   P["green"], P["blue"]]

fig, ax = plt.subplots(figsize=(10, 5), facecolor=P["bg"])
style_ax(ax, title="RandomForest Feature Importance",
         xlabel="Feature", ylabel="Importance")

bars = ax.bar(feat_names, feat_vals,
              color=bar_colors[:len(feat_names)],
              alpha=0.9, edgecolor=P["border"], linewidth=0.5)

for bar, val in zip(bars, feat_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}", ha="center", va="bottom",
            color=P["text"], fontsize=9, fontweight="bold")

ax.set_ylim(0, max(feat_vals) * 1.18)
ax.tick_params(axis="x", colors=P["text"], labelsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight",
            facecolor=P["bg"])
plt.close()
print("      Saved.")

# ── Image 4: Metrics Summary Card ─────────────────────────────────────────────
print("Generating metrics_summary.png…")

macro_p   = rep_pkl["macro avg"]["precision"]
macro_r   = rep_pkl["macro avg"]["recall"]
macro_f1  = rep_pkl["macro avg"]["f1-score"]
perfect   = sum(1 for n in names_sorted if rep_pkl[n]["f1-score"] == 1.0)
n_samples = cfg["training_info"]["total_samples"]

tiles = [
    ("Accuracy",         f"{acc_pkl*100:.2f}%",       P["green"]),
    ("Macro Precision",  f"{macro_p:.4f}",             P["blue"]),
    ("Macro Recall",     f"{macro_r:.4f}",             P["purple"]),
    ("Macro F1",         f"{macro_f1:.4f}",            P["amber"]),
    ("Perfect Classes",  f"{perfect}/22",              P["green"]),
    ("Train Samples",    f"{n_samples:,}",             P["sub"]),
    ("Augmentation",     f"{cfg['training_info']['augmentation_factor']}×", P["sub"]),
]

n = len(tiles)
fig = plt.figure(figsize=(14, 5), facecolor=P["bg"])
fig.suptitle("Model Performance Summary", color=P["text"],
             fontsize=16, fontweight="bold", y=0.98)

tile_w = 0.97 / n
for i, (label, value, color) in enumerate(tiles):
    ax = fig.add_axes([0.015 + i * tile_w, 0.12, tile_w - 0.018, 0.72])
    ax.set_facecolor(P["panel"])
    for sp in ax.spines.values():
        sp.set_edgecolor(color)
        sp.set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.62, value, transform=ax.transAxes,
            ha="center", va="center", fontsize=20,
            fontweight="bold", color=color)
    ax.text(0.5, 0.22, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color=P["sub"])

plt.savefig(OUT_DIR / "metrics_summary.png", dpi=150, bbox_inches="tight",
            facecolor=P["bg"])
plt.close()
print("      Saved.")

# ── Done ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DONE — images saved to benchmark/")
print("  confusion_matrix.png")
print("  per_class_metrics.png")
print("  feature_importance.png")
print("  metrics_summary.png")
print("=" * 60)
